# model-hop

One prompt. Every model. See who wins — in seconds.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Multi-Model](https://img.shields.io/badge/models-GPT%20%7C%20Claude%20%7C%20Gemini%20%7C%20Ollama-orange.svg)

<!-- ![demo](demo.gif) -->

## Why?
You shouldn't need a PhD in eval frameworks just to compare how different models handle your exact prompt. `model-hop` gives you a side-by-side latency, cost, and quality snapshot—ideal for prompt engineers, indie hackers, and anyone who wants the best output without the YAML.

## Features
- **Parallel Execution**: Fired all requests simultaneously using thread pools.
- **Auto-Detection**: Intelligently routes prompts to OpenAI, Anthropic, or Google based on model names.
- **Cost Estimation**: Built-in pricing table for major 2026 models.
- **Custom Endpoints**: Supports local models (Ollama, vLLM) via OpenAI-compatible base URLs.
- **Rich Output**: Beautiful terminal tables with sorting by latency, cost, or tokens.
- **Save to JSON**: Export full results for further analysis.

## Install
```bash
# Clone the repository
git clone https://github.com/amanhammadK/model-hop.git
cd model-hop

# Install recommended dependencies
pip install openai anthropic google-generativeai rich tiktoken
```

## Usage
```bash
# Basic usage (uses default models)
python model_hop.py "Write a tagline for a coffee brand for space travelers."

# Specify models
python model_hop.py "What is 2+2?" --models "gpt-4o,claude-3-haiku-20240307"

# Sort by cost and save results
python model_hop.py "Summarize the theory of relativity" --sort cost --save results.json

# Pipe from another command
cat prompt.txt | python model_hop.py --verbose
```

## API Configuration
Ensure your environment variables are set:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`

For custom models:
- Set `OPENAI_BASE_URL` or `OPENAI_BASE_URL_{MODELNAME}`.

## License
MIT
