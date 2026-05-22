import json
import time
import asyncio
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
from typing import TypedDict
from .data_agent import DataAgent
from .model_agent import ModelAgent

load_dotenv()

da = DataAgent()
ma = ModelAgent()
_logs  = []
_metrics = {}

#  Set up TinyLlama to explain results in plain language
llm = OllamaLLM(model="tinyllama")

#   Define the Pipeline State
class PipelineState(TypedDict):
    csv_path: str
    stage:    str
    metrics:  dict
    logs:     list

def _log(msg, level="info"):
    entry = {
        "time":  time.strftime("%H:%M:%S"),
        "msg":   msg,
        "level": level
    }
    _logs.append(entry)
    print(f"[{level.upper()}] {msg}")

def _generate_explanation(metrics: dict) -> str:
    """
    Uses OllamaLLM TinyLlama to generate
    a plain language explanation of results
    """
    try:
        task      = metrics.get("task", "unknown")
        model     = metrics.get("model_name", "unknown")
        accuracy  = metrics.get("accuracy", 0)
        rows      = metrics.get("rows", 0)
        features  = metrics.get("n_features", 0)
        shap      = metrics.get("shap", {})
        decision  = metrics.get("decision", "unknown")

        # Get top 3 features
        top_features = sorted(
            shap.items(),
            key=lambda x: -x[1]
        )[:3] if shap else []

        top_str = ", ".join([
            f"{f} ({v:.3f})"
            for f, v in top_features
        ]) if top_features else "not available"

        # Build prompt for OllamaLLM
        prompt = (
            f"Explain these machine learning results "
            f"in simple everyday language that anyone "
            f"can understand. Keep it to 3 sentences.\n\n"
            f"Results:\n"
            f"- Dataset: {rows} rows\n"
            f"- Features used: {features}\n"
            f"- Task type: {task}\n"
            f"- Best model: {model}\n"
            f"- Accuracy: {accuracy:.1%}\n"
            f"- Top features: {top_str}\n"
            f"- Decision: {decision}\n"
        )

        #  Use OllamaLLM to generate explanation
        explanation = llm.invoke(prompt)
        return explanation

    except Exception as e:
        _log(f"Explanation error: {e}", "warn")
        return (
            f"The system processed the dataset "
            f"and selected the best machine learning "
            f"model automatically."
        )

#   define LangChain Tools
@tool
def ingest_data(csv_path: str) -> str:
    """Ingests and cleans a CSV file.
    Input: full path to CSV file.
    Always call this tool first."""
    try:
        df, report = da.ingest(csv_path)
        msg = (
            f"SUCCESS: Loaded {len(df)} rows "
            f"and {df.shape[1]} columns. "
            f"Removed {report['duplicates']} "
            f"duplicates."
        )
        _log(msg)
        _metrics.update({
            "rows": len(df),
            "cols": df.shape[1]
        })
        return msg
    except Exception as e:
        _log(f"Ingest failed: {e}", "error")
        return f"ERROR: {e}"

@tool
def analyse_dataset(input: str = "run") -> str:
    """Analyses dataset for class balance.
    Input: just write run.
    Call this after ingest_data."""
    try:
        stats   = da.analyse()
        balance = stats["balance"]
        msg = (
            f"Class balance: {balance:.2f} | "
            f"Classes: {stats['n_classes']}"
        )
        if balance < 0.2:
            msg += " | WARNING: Class imbalance"
            _log(msg, "warn")
        else:
            _log(msg)
        _metrics.update(stats)
        return msg
    except Exception as e:
        _log(f"Analysis failed: {e}", "error")
        return f"ERROR: {e}"

@tool
def engineer_features(input: str = "run") -> str:
    """Selects best features using random forest.
    Input: just write run.
    Call this after analyse_dataset."""
    try:
        feats, imp = da.engineer_features()
        top = sorted(
            imp.items(), key=lambda x: -x[1]
        )[:3]
        top_str = " | ".join([
            f"{f}: {s:.3f}" for f, s in top
        ])
        msg = (
            f"Selected {len(feats)} features. "
            f"Top 3: {top_str}"
        )
        _log(msg)
        _metrics.update({
            "n_features": len(feats),
            "shap":       imp
        })
        return msg
    except Exception as e:
        _log(
            f"Feature engineering failed: {e}",
            "error"
        )
        return f"ERROR: {e}"

@tool
def train_best_model(input: str = "run") -> str:
    """Trains ML models and picks the best one.
    Input: just write run.
    Call this after engineer_features."""
    try:
        result = ma.train_and_select(da.X, da.y)
        msg = (
            f"Best model: {result['model_name']} | "
            f"Accuracy: {result['accuracy']:.3f} | "
            f"F1: {result['f1']:.3f}"
        )
        _log(msg)
        _metrics.update(result)
        return msg
    except Exception as e:
        _log(f"Training failed: {e}", "error")
        return f"ERROR: {e}"

@tool
def deploy_model(input: str = "run") -> str:
    """Deploys model if accuracy is good enough.
    Input: just write run.
    Always call this last."""
    try:
        acc = _metrics.get("accuracy", 0)
        if acc > 0.82:
            ma.save_model()
            decision = "AUTO DEPLOYED"
            _log(
                f"Model deployed - "
                f"confidence {acc:.1%}"
            )
        elif acc > 0.65:
            decision = "NEEDS HUMAN REVIEW"
            _log(
                f"Needs review - "
                f"confidence {acc:.1%}",
                "warn"
            )
        else:
            decision = "DEFERRED"
            _log(
                f"Deferred - "
                f"accuracy {acc:.1%} too low",
                "error"
            )

        _metrics["decision"]   = decision
        _metrics["confidence"] = round(acc, 4)

        #  Use OllamaLLM to generate explanation
        _log("Generating plain language explanation...")
        explanation = _generate_explanation(_metrics)
        _metrics["explanation"] = explanation
        _log(f"Explanation ready")

        return (
            f"Decision: {decision} | "
            f"Accuracy: {acc:.3f}"
        )
    except Exception as e:
        _log(f"Deploy failed: {e}", "error")
        return f"ERROR: {e}"

#   Define LangGraph Node Functions
def ingestion_node(
    state: PipelineState
) -> PipelineState:
    """Node 1 — reads and cleans the CSV file"""
    _log("Stage: ingestion")
    result = ingest_data.invoke(state["csv_path"])
    _log(f"Done: {result[:80]}")
    return {
        **state,
        "stage":   "ingestion",
        "metrics": _metrics.copy(),
        "logs":    _logs[-8:]
    }

def analysis_node(
    state: PipelineState
) -> PipelineState:
    """Node 2 — analyses the dataset"""
    _log("Stage: analysis")
    result = analyse_dataset.invoke("run")
    _log(f"Done: {result[:80]}")
    return {
        **state,
        "stage":   "analysis",
        "metrics": _metrics.copy(),
        "logs":    _logs[-8:]
    }

def features_node(
    state: PipelineState
) -> PipelineState:
    """Node 3 — selects the best features"""
    _log("Stage: features")
    result = engineer_features.invoke("run")
    _log(f"Done: {result[:80]}")
    return {
        **state,
        "stage":   "features",
        "metrics": _metrics.copy(),
        "logs":    _logs[-8:]
    }

def training_node(
    state: PipelineState
) -> PipelineState:
    """Node 4 — trains and selects best model"""
    _log("Stage: training")
    result = train_best_model.invoke("run")
    _log(f"Done: {result[:80]}")
    return {
        **state,
        "stage":   "training",
        "metrics": _metrics.copy(),
        "logs":    _logs[-8:]
    }

def deployment_node(
    state: PipelineState
) -> PipelineState:
    """Node 5 — deploys the best model"""
    _log("Stage: deployment")
    result = deploy_model.invoke("run")
    _log(f"Done: {result[:80]}")
    return {
        **state,
        "stage":   "deployment",
        "metrics": _metrics.copy(),
        "logs":    _logs[-8:]
    }

#   Build the LangGraph Pipeline
def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)

    # Add all agent nodes to the graph
    graph.add_node("ingestion",  ingestion_node)
    graph.add_node("analysis",   analysis_node)
    graph.add_node("features",   features_node)
    graph.add_node("training",   training_node)
    graph.add_node("deployment", deployment_node)

    # Define the flow between nodes
    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion",  "analysis")
    graph.add_edge("analysis",   "features")
    graph.add_edge("features",   "training")
    graph.add_edge("training",   "deployment")
    graph.add_edge("deployment", END)

    return graph.compile()

#   AgentOrchestrator uses LangGraph
class AgentOrchestrator:
    def __init__(self, broadcast=None):
        self.broadcast = broadcast
        self.tools = [
            ingest_data,
            analyse_dataset,
            engineer_features,
            train_best_model,
            deploy_model,
        ]
        # Build the LangGraph pipeline
        self.pipeline = build_pipeline()

    async def run(self, csv_path: str):
        _logs.clear()
        _metrics.clear()
        _log("AgentML pipeline starting...")
        await self._emit("starting")

        # Initial state for LangGraph
        initial_state: PipelineState = {
            "csv_path": csv_path,
            "stage":    "starting",
            "metrics":  {},
            "logs":     []
        }

        # Emit stage updates while pipeline runs
        stages = [
            "ingestion",
            "analysis",
            "features",
            "training",
            "deployment"
        ]

        for stage in stages:
            await self._emit(stage)
            await asyncio.sleep(0.3)

        try:
            #  run the full LangGraph pipeline
            final_state = self.pipeline.invoke(
                initial_state
            )
            _log("LangGraph pipeline completed!")

        except Exception as e:
            _log(
                f"Pipeline error: {e}",
                "error"
            )

        _log("All stages complete!")
        await self._emit("complete")

    async def _emit(self, stage):
        if not self.broadcast:
            return
        await self.broadcast(json.dumps({
            "stage":       stage,
            "logs":        _logs[-8:],
            "metrics":     _metrics,
            "decision":    _metrics.get(
                "decision", "—"
            ),
            "confidence":  _metrics.get(
                "confidence", 0
            ),
            "explanation": _metrics.get(
                "explanation", ""
            ),
        }))
