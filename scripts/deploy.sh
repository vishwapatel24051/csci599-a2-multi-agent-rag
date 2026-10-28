#!/usr/bin/env bash
# Deploy to Google Cloud Run. Idempotent: safe to re-run (re-running = rebuild + redeploy).
#
#   PROJECT_ID=my-project BILLING_ACCOUNT=XXXXXX-XXXXXX-XXXXXX ./scripts/deploy.sh
#
# What it does (each step skips work that is already done):
#   1. create the project (if missing) and link billing
#   2. enable Cloud Run, Cloud Build, Artifact Registry, Firestore, Secret Manager
#   3. create the Firestore (Native) database for conversation checkpoints
#   4. store the four API keys from .env in Secret Manager (never printed, never in the image)
#   5. create a dedicated runtime service account with least-privilege roles
#   6. build the image with Cloud Build and deploy it to Cloud Run
#
# Every gcloud call passes --project explicitly, so your active gcloud configuration is not changed.
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-east4}"            # Virginia: next to Pinecone Starter's AWS us-east-1
SERVICE="${SERVICE:-multi-agent-rag}"
REPO="${REPO:-containers}"
SA_NAME="${SA_NAME:-agent-lab-run}"
SA="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}"
SECRETS=(OPENAI_API_KEY NVIDIA_API_KEY PINECONE_API_KEY TAVILY_API_KEY)
P=(--project "$PROJECT_ID")

cd "$(dirname "$0")/.."
[[ -f .env ]] || { echo "missing .env (copy .env.example and fill in the keys)"; exit 1; }
# shellcheck disable=SC1091
set -a; source .env; set +a

step() { printf '\n==> %s\n' "$*"; }

step "1. project + billing"
if ! gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud projects create "$PROJECT_ID" --name="CSCI599 A2 multi-agent RAG"
fi
if [[ -n "${BILLING_ACCOUNT:-}" ]]; then
  gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT" >/dev/null
fi

step "2. APIs"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  firestore.googleapis.com secretmanager.googleapis.com iam.googleapis.com "${P[@]}"

step "3. Firestore (Native mode) for LangGraph checkpoints"
if ! gcloud firestore databases describe --database="(default)" "${P[@]}" >/dev/null 2>&1; then
  gcloud firestore databases create --location="$REGION" --type=firestore-native "${P[@]}"
fi

step "4. secrets (values come from .env and go straight to Secret Manager)"
for name in "${SECRETS[@]}"; do
  value="${!name:?$name missing from .env}"
  if ! gcloud secrets describe "$name" "${P[@]}" >/dev/null 2>&1; then
    printf '%s' "$value" | gcloud secrets create "$name" --data-file=- --replication-policy=automatic "${P[@]}" >/dev/null
  elif [[ "${UPDATE_SECRETS:-0}" == "1" ]]; then
    printf '%s' "$value" | gcloud secrets versions add "$name" --data-file=- "${P[@]}" >/dev/null
  fi
done

step "5. runtime service account (Firestore read/write + read these secrets only)"
if ! gcloud iam service-accounts describe "$SA" "${P[@]}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SA_NAME" --display-name="Agent Lab Cloud Run runtime" "${P[@]}"
fi
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$SA" \
  --role=roles/datastore.user --condition=None >/dev/null
for name in "${SECRETS[@]}"; do
  gcloud secrets add-iam-policy-binding "$name" --member="serviceAccount:$SA" \
    --role=roles/secretmanager.secretAccessor "${P[@]}" >/dev/null
done

step "6a. build the image with Cloud Build"
if ! gcloud artifacts repositories describe "$REPO" --location="$REGION" "${P[@]}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" "${P[@]}"
fi
gcloud builds submit --tag "$IMAGE" --region="$REGION" "${P[@]}"

step "6b. deploy to Cloud Run"
ENV_VARS="STATE_BACKEND=firestore,FIRESTORE_PROJECT_ID=${PROJECT_ID}"
for var in OPENAI_FAST_MODEL OPENAI_STRONG_MODEL OPENAI_EMBEDDING_MODEL NVIDIA_BASE_URL NVIDIA_STRONG_MODEL \
           NVIDIA_FAST_MODEL NVIDIA_RPM_LIMIT PINECONE_INDEX PINECONE_NAMESPACE RETRIEVAL_K CONTEXT_K MAX_HOPS; do
  [[ -n "${!var:-}" ]] && ENV_VARS+=",${var}=${!var}"
done
SECRET_REFS=""
for name in "${SECRETS[@]}"; do SECRET_REFS+="${name}=${name}:latest,"; done

gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --service-account "$SA" \
  --allow-unauthenticated \
  --memory 1Gi --cpu 1 --cpu-boost \
  --concurrency 10 \
  --timeout 600 \
  --min-instances 0 --max-instances 3 \
  --set-env-vars "$ENV_VARS" \
  --set-secrets "${SECRET_REFS%,}" \
  "${P[@]}"

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)' "${P[@]}")
step "deployed: $URL"
curl -fsS "$URL/health" && echo
