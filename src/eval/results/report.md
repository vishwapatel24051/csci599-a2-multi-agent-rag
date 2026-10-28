### RAGAS scores (mean over the 30-question set)

| Configuration | Faithfulness | Answer relevancy | Context precision | Context recall | Scored | Judge cost |
|---|---|---|---|---|---|---|
| auto (balanced routing) | 0.842 | 0.839 | 0.938 | 0.851 | 29/30 | $2.04 |
| OpenAI only | 0.896 | 0.827 | 0.921 | 0.885 | 29/30 | $2.10 |
| NVIDIA NIM only | 0.783 | 0.840 | 0.907 | 0.872 | 26/30 | $2.40 |
| auto (balanced routing), v1 baseline | 0.757 | 0.853 | 0.959 | 0.776 | 29/30 | $1.84 |

**Common subset** (the 25 questions scored in every configuration above):

| Configuration | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| auto (balanced routing) | 0.839 | 0.843 | 0.942 | 0.827 |
| OpenAI only | 0.900 | 0.835 | 0.915 | 0.893 |
| NVIDIA NIM only | 0.782 | 0.836 | 0.912 | 0.867 |
| auto (balanced routing), v1 baseline | 0.747 | 0.852 | 0.976 | 0.740 |

### How answers were grounded

| Configuration | Corpus retrieval | Web only | No sources | Extra agents after research | Supervisor guard fallbacks |
|---|---|---|---|---|---|
| auto (balanced routing) | 29 | 1 | 0 | 0 | 4 |
| OpenAI only | 29 | 1 | 0 | 0 | 3 |
| NVIDIA NIM only | 26 | 3 | 1 | 13 | 1 |
| auto (balanced routing), v1 baseline | 29 | 1 | 0 | 0 | 2 |

### Warm latency per query (seconds; p50 / p90)

| Configuration | n warm | TTFT (client) | Total (client) | Retrieval | LLM calls | MCP (web) | Server overhead | Network + queue |
|---|---|---|---|---|---|---|---|---|
| auto (balanced routing) | 30/30 | 7.3 / 17.0 | 16.7 / 30.3 | 0.3 / 0.9 | 14.3 / 23.3 | 0.0 / 1.7 | 0.2 / 7.7 | 0.1 / 0.1 |
| OpenAI only | 30/30 | 9.0 / 17.2 | 18.0 / 28.6 | 0.5 / 0.9 | 16.5 / 25.6 | 0.0 / 2.1 | 0.2 / 1.6 | 0.1 / 0.1 |
| NVIDIA NIM only | 30/30 | 6.1 / 20.1 | 15.4 / 46.7 | 0.4 / 0.8 | 13.4 / 31.7 | 0.0 / 0.1 | 0.2 / 2.6 | 0.1 / 0.2 |

**What happens before the first token** (client clock, p50 seconds from sending the request):

| Configuration | First byte | Supervisor decided | Agent started | First retrieval done | First answer token |
|---|---|---|---|---|---|
| auto (balanced routing) | 0.1 | 1.7 | 1.7 | 3.6 | 7.3 |
| OpenAI only | 0.1 | — | — | — | 9.0 |
| NVIDIA NIM only | 0.2 | 1.3 | 1.4 | 3.4 | 6.1 |

### Cost per query (USD, mean over the set)

| Configuration | LLM tokens in / out | LLM | Embeddings | Tavily | Cloud Run | Firestore | **Total** |
|---|---|---|---|---|---|---|---|
| auto (balanced routing) | 8,035 / 822 | 0.00336 | 0.0000004 | 0.0011 | 0.00067 | 0.000030 | **0.0051** |
| OpenAI only | 12,607 / 746 | 0.01863 | 0.0000007 | 0.0040 | 0.00053 | 0.000030 | **0.0232** |
| NVIDIA NIM only | 17,855 / 1,164 | 0.00000 | 0.0000003 | 0.0021 | 0.00137 | 0.000030 | **0.0035** |
| auto (balanced routing), v1 baseline | 7,320 / 886 | 0.00032 | 0.0000004 | 0.0003 | 0.00080 | 0.000030 | **0.0014** |

### Agents and models used

| Configuration | Agents (questions) | LLM calls by node:model | Errors |
|---|---|---|---|
| auto (balanced routing) | research (30) | research:gpt-6.1-sol ×18, research:nemotron-3-ultra-550b-a55b ×60, supervisor:gpt-6-luna ×56 | 5 |
| OpenAI only | research (30) | research:gpt-6.1-sol ×103, supervisor:gpt-6-luna ×57 | 3 |
| NVIDIA NIM only | research (17), research+code (1), research+code+summary (1), research+code+summary+data+research (1), research+summary (10) | code:nemotron-3-ultra-550b-a55b ×13, data:nemotron-3.5-lightning-30b-a3b ×7, research:nemotron-3-ultra-550b-a55b ×75, summary:nemotron-3-ultra-550b-a55b ×36, supervisor:nemotron-3.5-lightning-30b-a3b ×76, synthesize:nemotron-3-ultra-550b-a55b ×13 | 2 |
| auto (balanced routing), v1 baseline | research (30) | research:nemotron-3-ultra-550b-a55b ×80, supervisor:gpt-6-luna ×58 | 2 |

### Per question: auto (balanced routing)

| Question | Agent | Ctx | TTFT | Total | Retrieval | LLM | Cost | Faith. | Relev. | Prec. | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| react | research | 6 | 3.6 | 10.0 | 0.30 | 9.3 | 0.00039 | 0.85 | 0.92 | 0.65 | 1.00 |
| reflexion | research | 9 | 11.2 | 18.1 | 0.00 | 14.0 | 0.00028 | 0.80 | 0.87 | 0.99 | 0.67 |
| rewoo_vs_llmcompiler | research | 5 | 7.5 | 14.9 | 0.54 | 13.9 | 0.00030 | 0.35 | 0.89 | 1.00 | 0.00 |
| self_consistency | research | 5 | 13.0 | 25.5 | 0.21 | 17.4 | 0.00026 | 0.67 | 0.85 | 1.00 | 1.00 |
| tot | research | 5 | 3.3 | 16.3 | 0.48 | 14.7 | 0.00027 | 1.00 | 0.75 | 1.00 | 1.00 |
| toolformer | research | 5 | 8.0 | 14.0 | 0.85 | 5.5 | 0.00048 | 1.00 | 0.79 | 1.00 | 0.50 |
| selfrag_vs_crag | research | 10 | 21.8 | 30.3 | 0.24 | 15.7 | 0.00033 | 0.72 | 0.91 | 0.88 | 1.00 |
| adaptive_rag | research | 7 | 18.1 | 46.2 | 0.55 | 37.0 | 0.00034 | 0.83 | 0.79 | 0.92 | 1.00 |
| raptor | research | 7 | 16.9 | 28.7 | 0.85 | 18.8 | 0.00013 | 0.76 | 0.86 | 0.82 | 1.00 |
| graphrag | research | 8 | 6.6 | 12.0 | 0.94 | 10.8 | 0.00028 | 0.90 | 0.88 | 0.75 | 1.00 |
| lost_middle | research | 5 | 2.9 | 8.3 | 0.24 | 7.8 | 0.00028 | 0.92 | 0.78 | 0.83 | 1.00 |
| dpr | research | 10 | 6.2 | 41.0 | 0.30 | 40.5 | 0.00044 | 0.85 | 0.91 | 0.77 | 0.33 |
| hyde | research | 5 | 11.7 | 25.7 | 0.00 | 22.7 | 0.00029 | 1.00 | 0.89 | 1.00 | 1.00 |
| frugalgpt | research | 5 | 4.0 | 9.9 | 0.51 | 9.0 | 0.00042 | 0.82 | 0.72 | 1.00 | 1.00 |
| routellm | research | 5 | 17.9 | 30.2 | 1.29 | 28.6 | 0.00013 | 0.71 | 0.77 | 1.00 | 0.67 |
| mixtral | research | 0 | 12.3 | 15.8 | 0.24 | 8.1 | 0.00030 | — | — | — | — |
| metagpt | research | 5 | 5.7 | 14.0 | 0.23 | 12.9 | 0.00025 | 0.81 | 0.85 | 1.00 | 1.00 |
| debate | research | 7 | 7.2 | 17.6 | 0.50 | 14.8 | 0.00026 | 0.82 | 0.89 | 1.00 | 1.00 |
| more_agents | research | 5 | 7.7 | 13.6 | 0.45 | 11.2 | 0.00040 | 0.65 | 0.72 | 1.00 | 1.00 |
| moa | research | 5 | 4.5 | 18.7 | 0.23 | 18.1 | 0.00044 | 0.94 | 0.80 | 1.00 | 1.00 |
| magentic_one | research | 5 | 14.3 | 22.3 | 0.29 | 21.8 | 0.00899 | 1.00 | 0.93 | 1.00 | 0.67 |
| mast | research | 7 | 7.9 | 18.7 | 0.50 | 17.9 | 0.00012 | 0.95 | 0.81 | 0.95 | 0.33 |
| llm_judge | research | 5 | 6.9 | 17.1 | 0.62 | 16.2 | 0.00013 | 0.59 | 0.85 | 0.64 | 0.50 |
| ragas | research | 5 | 5.4 | 9.5 | 0.32 | 9.0 | 0.00029 | 1.00 | 0.82 | 1.00 | 1.00 |
| lg_checkpointer | research | 13 | 4.1 | 11.1 | 0.24 | 10.5 | 0.02181 | 0.87 | 0.82 | 1.00 | 1.00 |
| lg_command | research | 7 | 7.3 | 14.0 | 0.52 | 13.2 | 0.01397 | 0.96 | 0.88 | 1.00 | 1.00 |
| mcp_transports | research | 5 | 4.2 | 14.5 | 0.22 | 14.0 | 0.02547 | 0.65 | 0.81 | 1.00 | 1.00 |
| mcp_server_features | research | 5 | 3.7 | 10.0 | 0.20 | 9.5 | 0.00737 | 1.00 | 0.84 | 1.00 | 1.00 |
| a2a_agent_card | research | 5 | 9.1 | 19.3 | 0.00 | 16.9 | 0.00709 | 1.00 | 0.87 | 1.00 | 1.00 |
| a2a_vs_mcp | research | 5 | 5.9 | 20.2 | 0.28 | 19.7 | 0.00930 | 1.00 | 0.87 | 1.00 | 1.00 |

### Cold starts (min-instances = 0)

| Probe (UTC) | Confirmed new instance | First byte | TTFT | Total | Container + imports | App startup |
|---|---|---|---|---|---|---|
| 2026-10-27T21:08:38+00:00 | yes | 20.6 | 29.3 | 39.2 | 7.3 | 12.4 |
| 2026-10-27T21:28:50+00:00 | yes | 8.3 | 24.4 | 83.1 | 3.4 | 4.3 |
| 2026-10-27T21:49:45+00:00 | yes | 7.8 | 20.8 | 223.8 | 3.2 | 4.2 |
| 2026-10-27T23:53:44+00:00 | yes | 6.6 | 12.2 | 15.4 | 2.6 | 3.5 |
