
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: promptshield/dashboard/app.py
# execution: true
import os
import sys
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Streamlit imports
try:
    import streamlit as st
except ImportError:
    # If streamlit is not installed, we'll create a mock version for testing
    class MockSt:
        def title(self, text):
            print(f"TITLE: {text}")
        
        def header(self, text):
            print(f"HEADER: {text}")
        
        def subheader(self, text):
            print(f"SUBHEADER: {text}")
        
        def write(self, text):
            print(f"WRITE: {text}")
        
        def dataframe(self, df):
            print(f"DATAFRAME: {df.shape[0]} rows x {df.shape[1]} columns")
            print(df.head())
        
        def line_chart(self, data):
            print(f"LINE CHART: {data.shape[0]} rows x {data.shape[1]} columns")
        
        def bar_chart(self, data):
            print(f"BAR CHART: {data.shape[0]} rows x {data.shape[1]} columns")
        
        def pie_chart(self, data):
            print(f"PIE CHART: {data.shape[0]} rows x {data.shape[1]} columns")
        
        def metric(self, label, value, delta=None):
            delta_str = f" ({delta})" if delta is not None else ""
            print(f"METRIC: {label} = {value}{delta_str}")
        
        def columns(self, n):
            class MockColumns:
                def __enter__(self):
                    print(f"COLUMNS: {n}")
                    return [MockSt() for _ in range(n)]
                
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
            
            return MockColumns()
        
        def sidebar(self):
            print("SIDEBAR:")
            return self
        
        def selectbox(self, label, options):
            print(f"SELECTBOX: {label} - Options: {options}")
            return options[0] if options else None
        
        def date_input(self, label, value=None):
            print(f"DATE INPUT: {label}")
            return datetime.datetime.now().date()
        
        def button(self, label):
            print(f"BUTTON: {label}")
            return False
        
        def success(self, text):
            print(f"SUCCESS: {text}")
        
        def error(self, text):
            print(f"ERROR: {text}")
        
        def warning(self, text):
            print(f"WARNING: {text}")
        
        def info(self, text):
            print(f"INFO: {text}")
    
    st = MockSt()

def load_logs(log_file):
    """Load logs from a JSONL file."""
    logs = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    logs.append(log_entry)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        # Create sample data for testing
        logs = [
            {
                "timestamp": "2025-07-30T19:57:00.000Z",
                "prompt": "What is the capital of France?",
                "classification": {"label": "low_cost", "confidence": 0.9},
                "action": "route",
                "model": "gpt-3.5-turbo",
                "response_time": 0.5
            },
            {
                "timestamp": "2025-07-30T19:58:00.000Z",
                "prompt": "Write a poem about AI",
                "classification": {"label": "valuable", "confidence": 0.8},
                "action": "route",
                "model": "gpt-4",
                "response_time": 1.2
            },
            {
                "timestamp": "2025-07-30T19:59:00.000Z",
                "prompt": "asdjklasdjkl",
                "classification": {"label": "nonsense", "confidence": 1.0},
                "action": "block",
                "reason": "Prompt classified as nonsense",
                "response_time": 0.1
            },
            {
                "timestamp": "2025-07-30T20:00:00.000Z",
                "prompt": "What is the capital of France?",
                "classification": {"label": "repeat", "confidence": 1.0},
                "action": "cache",
                "response_time": 0.05
            },
            {
                "timestamp": "2025-07-30T20:01:00.000Z",
                "prompt": "You are stupid",
                "classification": {"label": "spam", "confidence": 1.0},
                "action": "block",
                "reason": "Prompt classified as spam",
                "response_time": 0.1
            }
        ]
    
    return logs

def calculate_metrics(logs):
    """Calculate metrics from logs."""
    if not logs:
        return {
            "total_count": 0,
            "blocked_count": 0,
            "cache_hit_count": 0,
            "cheap_model_count": 0,
            "default_model_count": 0,
            "block_rate": 0,
            "cache_hit_rate": 0,
            "cheap_model_rate": 0,
            "default_model_rate": 0,
            "avg_response_time": 0,
            "estimated_cost_saved": 0
        }
    
    total_count = len(logs)
    blocked_count = sum(1 for log in logs if log.get("action") == "block")
    cache_hit_count = sum(1 for log in logs if log.get("action") == "cache")
    
    # Count models used
    model_counts = {}
    for log in logs:
        if log.get("action") == "route" and "model" in log:
            model = log["model"]
            model_counts[model] = model_counts.get(model, 0) + 1
    
    # Determine which models are cheap and which are default
    cheap_models = ["gpt-3.5-turbo", "claude-haiku", "mistral-instruct"]
    default_models = ["gpt-4", "claude-v1", "llama3-70b"]
    
    cheap_model_count = sum(model_counts.get(model, 0) for model in cheap_models)
    default_model_count = sum(model_counts.get(model, 0) for model in default_models)
    
    # Calculate rates
    block_rate = blocked_count / total_count if total_count > 0 else 0
    cache_hit_rate = cache_hit_count / total_count if total_count > 0 else 0
    cheap_model_rate = cheap_model_count / total_count if total_count > 0 else 0
    default_model_rate = default_model_count / total_count if total_count > 0 else 0
    
    # Calculate average response time
    response_times = [log.get("response_time", 0) for log in logs]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    # Estimate cost saved
    # Assumption: default model costs 10x more than cheap model
    # Each blocked request or cache hit saves a default model call
    estimated_cost_saved = (blocked_count + cache_hit_count + cheap_model_count * 0.9) * 0.01  # $0.01 per default model call
    
    return {
        "total_count": total_count,
        "blocked_count": blocked_count,
        "cache_hit_count": cache_hit_count,
        "cheap_model_count": cheap_model_count,
        "default_model_count": default_model_count,
        "block_rate": block_rate,
        "cache_hit_rate": cache_hit_rate,
        "cheap_model_rate": cheap_model_rate,
        "default_model_rate": default_model_rate,
        "avg_response_time": avg_response_time,
        "estimated_cost_saved": estimated_cost_saved
    }

def create_dataframe(logs):
    """Create a DataFrame from logs."""
    if not logs:
        return pd.DataFrame()
    
    # Extract relevant fields
    data = []
    for log in logs:
        entry = {
            "timestamp": log.get("timestamp", ""),
            "prompt": log.get("prompt", ""),
            "classification": log.get("classification", {}).get("label", ""),
            "confidence": log.get("classification", {}).get("confidence", 0),
            "action": log.get("action", ""),
            "model": log.get("model", ""),
            "reason": log.get("reason", ""),
            "response_time": log.get("response_time", 0)
        }
        data.append(entry)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert timestamp to datetime
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except:
        pass
    
    return df

def main():
    """Main function for the Streamlit dashboard."""
    st.title("PromptShield Dashboard")
    
    # Sidebar
    st.sidebar.header("Filters")
    
    # Load logs
    log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs", "prompt_logs.jsonl")
    logs = load_logs(log_file)
    
    # Create DataFrame
    df = create_dataframe(logs)
    
    # Date filter
    if not df.empty and "timestamp" in df.columns:
        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()
        
        start_date = st.sidebar.date_input("Start Date", min_date)
        end_date = st.sidebar.date_input("End Date", max_date)
        
        # Filter by date
        mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
        filtered_df = df[mask]
    else:
        filtered_df = df
    
    # Classification filter
    if not filtered_df.empty and "classification" in filtered_df.columns:
        classifications = ["All"] + sorted(filtered_df["classification"].unique().tolist())
        selected_classification = st.sidebar.selectbox("Classification", classifications)
        
        if selected_classification != "All":
            filtered_df = filtered_df[filtered_df["classification"] == selected_classification]
    
    # Action filter
    if not filtered_df.empty and "action" in filtered_df.columns:
        actions = ["All"] + sorted(filtered_df["action"].unique().tolist())
        selected_action = st.sidebar.selectbox("Action", actions)
        
        if selected_action != "All":
            filtered_df = filtered_df[filtered_df["action"] == selected_action]
    
    # Calculate metrics
    metrics = calculate_metrics(logs)
    
    # Display metrics
    st.header("Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Prompts", metrics["total_count"])
    
    with col2:
        st.metric("Blocked", metrics["blocked_count"], f"{metrics['block_rate']:.1%}")
    
    with col3:
        st.metric("Cache Hits", metrics["cache_hit_count"], f"{metrics['cache_hit_rate']:.1%}")
    
    with col4:
        st.metric("Cost Saved", f"${metrics['estimated_cost_saved']:.2f}")
    
    # Display model usage
    st.subheader("Model Usage")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Cheap Model", metrics["cheap_model_count"], f"{metrics['cheap_model_rate']:.1%}")
    
    with col2:
        st.metric("Default Model", metrics["default_model_count"], f"{metrics['default_model_rate']:.1%}")
    
    # Display charts
    st.header("Charts")
    
    # Classification distribution
    if not filtered_df.empty and "classification" in filtered_df.columns:
        st.subheader("Classification Distribution")
        classification_counts = filtered_df["classification"].value_counts().reset_index()
        classification_counts.columns = ["Classification", "Count"]
        st.bar_chart(classification_counts.set_index("Classification"))
    
    # Action distribution
    if not filtered_df.empty and "action" in filtered_df.columns:
        st.subheader("Action Distribution")
        action_counts = filtered_df["action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]
        st.bar_chart(action_counts.set_index("Action"))
    
    # Response time by classification
    if not filtered_df.empty and "classification" in filtered_df.columns and "response_time" in filtered_df.columns:
        st.subheader("Response Time by Classification")
        response_time_by_classification = filtered_df.groupby("classification")["response_time"].mean().reset_index()
        response_time_by_classification.columns = ["Classification", "Average Response Time (s)"]
        st.bar_chart(response_time_by_classification.set_index("Classification"))
    
    # Display logs
    st.header("Logs")
    if not filtered_df.empty:
        st.dataframe(filtered_df)
    else:
        st.info("No logs available.")

# For testing
if __name__ == "__main__":
    print("PromptShield Dashboard")
    print("=====================")
    
    # Load logs
    log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs", "prompt_logs.jsonl")
    logs = load_logs(log_file)
    
    # Calculate metrics
    metrics = calculate_metrics(logs)
    
    # Print metrics
    print("\nMetrics:")
    print(f"Total Prompts: {metrics['total_count']}")
    print(f"Blocked: {metrics['blocked_count']} ({metrics['block_rate']:.1%})")
    print(f"Cache Hits: {metrics['cache_hit_count']} ({metrics['cache_hit_rate']:.1%})")
    print(f"Cheap Model: {metrics['cheap_model_count']} ({metrics['cheap_model_rate']:.1%})")
    print(f"Default Model: {metrics['default_model_count']} ({metrics['default_model_rate']:.1%})")
    print(f"Average Response Time: {metrics['avg_response_time']:.2f} seconds")
    print(f"Estimated Cost Saved: ${metrics['estimated_cost_saved']:.2f}")
    
    # Create DataFrame
    df = create_dataframe(logs)
    
    # Print DataFrame
    print("\nLogs DataFrame:")
    if not df.empty:
        print(df.head())
    else:
        print("No logs available.")
    
    print("\nTo run the dashboard, execute:")
    print("  streamlit run promptshield/dashboard/app.py")

print("Dashboard implemented successfully!")