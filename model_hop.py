#!/usr/bin/env python3
import argparse
import os
import sys
import time
import concurrent.futures
import json
import textwrap
from datetime import datetime
import math

# Attempt to import optional libraries
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Pricing Table (USD per 1M tokens) - Approx 2026 Prices
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
}

DEFAULT_MODELS = "gpt-4o,claude-sonnet-4-20250514,gemini-2.5-pro"

class ModelHop:
    def __init__(self, timeout=60):
        self.timeout = timeout
        self.results = []

    def estimate_tokens(self, text, model_name):
        if tiktoken and "gpt" in model_name:
            try:
                encoding = tiktoken.encoding_for_model(model_name)
                return len(encoding.encode(text))
            except:
                pass
        # Heuristic fallback
        return math.ceil(len(text.split()) * 1.3)

    def get_provider(self, model_id):
        if model_id.startswith(("gpt-", "o1", "o3", "o4")):
            return "openai"
        elif model_id.startswith("claude-"):
            return "anthropic"
        elif model_id.startswith("gemini-"):
            return "google"
        elif "/" in model_id:
            return "custom"
        return "unknown"

    def call_model(self, model_id, prompt):
        provider = self.get_provider(model_id)
        start_time = time.time()
        result = {
            "model": model_id,
            "response": "",
            "latency": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost": 0.0,
            "error": None
        }

        try:
            if provider == "openai":
                if not openai: raise ImportError("openai library not installed")
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout
                )
                result["response"] = resp.choices[0].message.content
                result["tokens_in"] = resp.usage.prompt_tokens
                result["tokens_out"] = resp.usage.completion_tokens

            elif provider == "anthropic":
                if not anthropic: raise ImportError("anthropic library not installed")
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                resp = client.messages.create(
                    model=model_id,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout
                )
                result["response"] = resp.content[0].text
                result["tokens_in"] = resp.usage.input_tokens
                result["tokens_out"] = resp.usage.output_tokens

            elif provider == "google":
                if not genai: raise ImportError("google-generativeai library not installed")
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                model = genai.GenerativeModel(model_id)
                resp = model.generate_content(prompt)
                result["response"] = resp.text
                # Gemini usage metadata varies; fallback to estimation if missing
                result["tokens_in"] = getattr(resp.usage_metadata, 'prompt_token_count', self.estimate_tokens(prompt, model_id))
                result["tokens_out"] = getattr(resp.usage_metadata, 'candidates_token_count', self.estimate_tokens(resp.text, model_id))

            elif provider == "custom":
                # Handle custom OpenAI-compatible endpoint
                short_name = model_id.split("/")[-1]
                base_url = os.getenv(f"OPENAI_BASE_URL_{short_name.upper()}") or os.getenv("OPENAI_BASE_URL")
                if not base_url: raise ValueError(f"Custom model requires OPENAI_BASE_URL or OPENAI_BASE_URL_{short_name.upper()}")
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "no-key"), base_url=base_url)
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout
                )
                result["response"] = resp.choices[0].message.content
                result["tokens_in"] = resp.usage.prompt_tokens
                result["tokens_out"] = resp.usage.completion_tokens
            
            else:
                raise ValueError(f"Unknown provider for model: {model_id}")

            result["latency"] = (time.time() - start_time) * 1000
            
            # Calculate Cost
            if model_id in PRICING:
                prices = PRICING[model_id]
                result["cost"] = (result["tokens_in"] / 1_000_000 * prices["input"]) + \
                                 (result["tokens_out"] / 1_000_000 * prices["output"])
            else:
                result["cost"] = -1.0 # Unknown

        except Exception as e:
            result["error"] = str(e)
            result["response"] = f"ERROR: {str(e)}"
            result["latency"] = (time.time() - start_time) * 1000

        return result

    def run(self, prompt, models_list):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.call_model, m.strip(), prompt) for m in models_list]
            self.results = [f.result() for f in concurrent.futures.as_completed(futures)]

def display_results(results, sort_by="latency", verbose=False):
    # Sorting
    if sort_by == "latency":
        results.sort(key=lambda x: x["latency"])
    elif sort_by == "cost":
        results.sort(key=lambda x: x["cost"] if x["cost"] >= 0 else float('inf'))
    elif sort_by == "tokens":
        results.sort(key=lambda x: x["tokens_in"] + x["tokens_out"])

    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="Model Comparison Results")
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("Model", style="magenta")
        table.add_column("Latency", justify="right", style="green")
        table.add_column("Tokens (in/out)", justify="right")
        table.add_column("Cost (USD)", justify="right", style="yellow")
        table.add_column("First words", style="white")

        for i, r in enumerate(results, 1):
            cost_str = f"${r['cost']:.4f}" if r["cost"] >= 0 else "Unknown"
            if r["error"]:
                table.add_row(str(i), r["model"], f"{r['latency']:.0f}ms", "N/A", "N/A", f"[red]{r['response'][:100]}...[/red]")
            else:
                snippet = textwrap.shorten(r["response"], width=200, placeholder="...")
                table.add_row(
                    str(i), 
                    r["model"], 
                    f"{r['latency']:.0f}ms", 
                    f"{r['tokens_in']}/{r['tokens_out']}", 
                    cost_str, 
                    snippet
                )
        console.print(table)
    else:
        # Plain text fallback
        header = f"{'#':<3} {'Model':<25} {'Latency':<10} {'Tokens':<15} {'Cost':<10} {'First words'}"
        print(header)
        print("-" * len(header))
        for i, r in enumerate(results, 1):
            cost_str = f"${r['cost']:.4f}" if r["cost"] >= 0 else "N/A"
            snippet = (r["response"][:100] + "...") if len(r["response"]) > 100 else r["response"]
            tokens = f"{r['tokens_in']}/{r['tokens_out']}"
            print(f"{i:<3} {r['model']:<25} {r['latency']:4.0f}ms {tokens:<15} {cost_str:<10} {snippet}")

    if verbose:
        print("\n" + "="*50)
        print("FULL RESPONSES")
        print("="*50)
        for r in results:
            print(f"\n[{r['model'].upper()}]")
            print("-" * (len(r['model']) + 2))
            print(r["response"])
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="model-hop: Compare multiple LLM prompts simultaneously.")
    parser.add_argument("prompt", nargs="?", help="The prompt to send to models. If empty, reads from stdin.")
    parser.add_argument("--models", default=DEFAULT_MODELS, help=f"Comma-separated list of model IDs. Default: {DEFAULT_MODELS}")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds per request (default: 60)")
    parser.add_argument("--sort", choices=["latency", "cost", "tokens"], default="latency", help="Sort results by metric (default: latency)")
    parser.add_argument("--save", help="Save full results to a JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full model responses")
    
    args = parser.parse_args()

    # Get prompt from arg or stdin
    prompt = args.prompt
    if not prompt:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            parser.print_help()
            sys.exit(0)

    models_list = args.models.split(",")
    
    # Check for missing keys and skip models
    active_models = []
    for m in models_list:
        m = m.strip()
        provider = ModelHop().get_provider(m)
        key_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GEMINI_API_KEY", "custom": "OPENAI_API_KEY"}
        env_key = key_map.get(provider)
        if env_key and not os.getenv(env_key):
            print(f"Skipping model: {m} ({env_key} not set)", file=sys.stderr)
            continue
        active_models.append(m)

    if not active_models:
        print("Error: No models available (missing API keys).", file=sys.stderr)
        sys.exit(1)

    hopper = ModelHop(timeout=args.timeout)
    print(f"Firing prompt to {len(active_models)} models...")
    hopper.run(prompt, active_models)
    
    display_results(hopper.results, sort_by=args.sort, verbose=args.verbose)

    if args.save:
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "results": hopper.results
        }
        with open(args.save, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {args.save}")

if __name__ == "__main__":
    main()
