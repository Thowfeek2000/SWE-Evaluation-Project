# import os
# import logging
# import time
# from typing import Any, Dict, List, Optional

# import dspy
# import google.generativeai as genai
# from dspy import Signature, InputField, OutputField


# # -----------------------------------------------------
# # 1. Logging Configuration
# # -----------------------------------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s"
# )
# logger = logging.getLogger(__name__)


# # -----------------------------------------------------
# # 2. GeminiLM CLASS
# # -----------------------------------------------------
# class GeminiLM(dspy.LM):
#     """DSPy-compatible LLM wrapper for Google Gemini models."""

#     def __init__(self, model: str, api_key: str):
#         super().__init__(model)
#         genai.configure(api_key=api_key)
#         self._model = genai.GenerativeModel(model)

#     def _retry_request(self, func, *args, retries: int = 3, delay: int = 2, **kwargs) -> Any:
#         """Retry mechanism for Gemini API calls."""
#         for attempt in range(retries):
#             try:
#                 return func(*args, **kwargs)
#             except Exception as e:
#                 logger.warning("Gemini API call failed (attempt %d/%d): %s", attempt + 1, retries, e)
#                 if attempt < retries - 1:
#                     time.sleep(delay)
#         raise RuntimeError("Gemini API call failed after multiple retries.")

#     def __call__(self, *args, **kwargs) -> List[str]:
#         """Handle DSPy prompt/message/query inputs."""
#         prompt: Optional[str] = kwargs.get("prompt")
#         messages: Optional[List[Dict[str, Any]]] = kwargs.get("messages")
#         query: Optional[str] = kwargs.get("query")

#         if prompt:
#             response = self._retry_request(self._model.generate_content, prompt)
#             return [response.text]

#         elif messages:
#             gemini_messages = []
#             for m in messages:
#                 role = "user" if m.get("role") == "system" else m.get("role", "user")
#                 content = m.get("content", "")
#                 gemini_messages.append({"role": role, "parts": [{"text": content}]})

#             response = self._retry_request(self._model.generate_content, gemini_messages)
#             return [response.text]

#         elif query:
#             response = self._retry_request(self._model.generate_content, query)
#             return [response.text]

#         else:
#             raise ValueError("No valid input provided to GeminiLM")


# # -----------------------------------------------------
# # 3. CONFIGURE DSPy with Gemini (API Key via Env Var)
# # -----------------------------------------------------
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise EnvironmentError("Missing environment variable: GEMINI_API_KEY")

# logger.info("Configuring DSPy with GeminiLM...")
# dspy.configure(lm=GeminiLM("gemini-2.5-flash", GEMINI_API_KEY))
# logger.info("DSPy configured successfully.")


# # -----------------------------------------------------
# # 4. SIGNATURES
# # -----------------------------------------------------
# class QueryUnderstanding(Signature):
#     """Extracts intent and entities from user query."""
#     query = InputField(desc="The natural language query from the user.")
#     intent = OutputField(desc="High-level intent, e.g., FindPendingFollowUps, GenerateProgressReport.")
#     entities = OutputField(desc="Extracted structured entities in JSON format.")


# class ToolSelection(Signature):
#     """Maps intents to tools."""
#     intent = InputField()
#     selected_tools = OutputField(desc="List of relevant tools to use from AVAILABLE_TOOLS.")


# class PlanGeneration(Signature):
#     """Generates execution plan based on query and tools."""
#     query = InputField()
#     selected_tools = InputField()
#     plan = OutputField(desc="Step-by-step JSON plan with tool calls and processing instructions.")


# # -----------------------------------------------------
# # 5. AVAILABLE TOOLS
# # -----------------------------------------------------
# AVAILABLE_TOOLS: List[str] = [
#     "get_user_identities",
#     "list_open_issues",
#     "list_unread_emails",
#     "list_recent_chats",
#     "get_project_progress",
#     "get_code_coverage_goal_progress"
# ]


# # -----------------------------------------------------
# # 6. TOOL SELECTION MODULE (Configurable Mapping)
# # -----------------------------------------------------
# class ToolSelectionModule(dspy.Module):
#     """Selects tools for an intent based on config mapping."""

#     def __init__(self, available_tools: List[str], intent_tool_map: Dict[str, List[str]]):
#         super().__init__()
#         self.available_tools = available_tools
#         self.intent_tool_map = intent_tool_map
#         self.predict = dspy.Predict(ToolSelection)

#     def forward(self, intent: str) -> dspy.Prediction:
#         result = self.predict(intent=intent)
#         selected = self.intent_tool_map.get(intent, [])
#         selected = [t for t in selected if t in self.available_tools]
#         return dspy.Prediction(selected_tools=selected)


# # -----------------------------------------------------
# # 7. PLAN GENERATION MODULE (Parameter Resolution)
# # -----------------------------------------------------
# import json

# class PlanGenerationModule(dspy.Module):
#     """Generates execution plan, resolving placeholders."""

#     def __init__(self):
#         super().__init__()
#         self.predict = dspy.Predict(PlanGeneration)

#     def forward(self, query: str, selected_tools: List[str], entities: Any) -> dspy.Prediction:
#         # Ensure entities is a dict
#         if isinstance(entities, str):
#             try:
#                 entities = json.loads(entities)
#             except Exception:
#                 entities = {}
#         elif not isinstance(entities, dict):
#             entities = {}

#         intent = None
#         q_lower = query.lower()

#         if "progress" in q_lower and "project" in q_lower:
#             intent = "GenerateProgressReport"
#         elif "code-coverage" in q_lower:
#             intent = "GetCodeCoverageProgress"
#         elif "follow up" in q_lower:
#             intent = "FindPendingFollowUps"

#         plan = []

#         if intent == "FindPendingFollowUps":
#             plan = [
#                 {"step": 1, "description": "Find user identities", "tool_call": f"get_user_identities('{entities.get('user', 'unknown')}')"},
#                 {"step": 2, "description": "Fetch open issues", "tool_call": "list_open_issues(tool_id, user_identity)"},
#                 {"step": 3, "description": "Fetch unread emails", "tool_call": "list_unread_emails(user_identity)"},
#                 {"step": 4, "description": "Fetch recent chats", "tool_call": f"list_recent_chats(tool_id, user_identity, '{entities.get('date', 'today')}')"},
#                 {"step": 5, "description": "Consolidate", "processing": "deduplicate_and_consolidate"},
#                 {"step": 6, "description": "Format JSON", "output_format": "json_with_source_links"}
#             ]
#         elif intent == "GenerateProgressReport":
#             plan = [
#                 {"step": 1, "description": "Get project progress", "tool_call": f"get_project_progress('{entities.get('project', 'unknown')}')"},
#                 {"step": 2, "description": "Format report", "processing": "format_project_report"},
#                 {"step": 3, "description": "Generate document", "output_format": "document_report"}
#             ]
#         elif intent == "GetCodeCoverageProgress":
#             plan = [
#                 {"step": 1, "description": "Retrieve code coverage", "tool_call": "get_code_coverage_goal_progress()"},
#                 {"step": 2, "description": "Summarize", "output_format": "simple_text_summary"}
#             ]

#         if plan:
#             return dspy.Prediction(plan=plan)

#         # fallback to LLM if not found
#         return self.predict(query=query, selected_tools=selected_tools)


# # -----------------------------------------------------
# # 8. PIPELINE MODULE
# # -----------------------------------------------------
# class QueryPipeline(dspy.Module):
#     """Pipeline to process queries into plans."""

#     def __init__(self, intent_tool_map: Dict[str, List[str]]):
#         super().__init__()
#         self.understand = dspy.Predict(QueryUnderstanding)
#         self.select_tools = ToolSelectionModule(AVAILABLE_TOOLS, intent_tool_map)
#         self.generate_plan = PlanGenerationModule()

#     def forward(self, query: str) -> Dict[str, Any]:
#         understood = self.understand(query=query)
#         tools = self.select_tools(intent=understood.intent)
#         plan = self.generate_plan(query=query, selected_tools=tools.selected_tools, entities=understood.entities)
#         return {
#             "query": query,
#             "intent": understood.intent,
#             "entities": understood.entities,
#             "selected_tools": tools.selected_tools,
#             "plan": plan.plan
#         }


# # -----------------------------------------------------
# # 9. EXECUTION
# # -----------------------------------------------------
# if __name__ == "__main__":
#     #  Configurable mapping
#     INTENT_TOOL_MAP = {
#         "FindPendingFollowUps": ["get_user_identities", "list_open_issues", "list_unread_emails", "list_recent_chats"],
#         "GenerateProgressReport": ["get_project_progress"],
#         "GetCodeCoverageProgress": ["get_code_coverage_goal_progress"],
#     }

#     queries = [
#         "Find my pending open follow ups",
#         "Generate a report on the current progress on Project X.",
#         "Find the progress on the current code-coverage goal for the company"
#     ]

#     query_pipeline = QueryPipeline(INTENT_TOOL_MAP)

#     logger.info("Running pipeline...")
#     for q in queries:
#       try:
#         result = query_pipeline(query=q)
#         print("\nQuery:", q)
#         print("Intent:", result.get("intent"))
#         print("Entities:", result.get("entities"))
#         print("Tools:", result.get("selected_tools"))
#         plan_data = result.get("plan")
#         if hasattr(plan_data, "to_dict"):
#             plan_data = plan_data.to_dict()
#         print("Plan:", plan_data)
#         print("="*50)
#       except Exception as e:
#         print(f"Error for query '{q}': {e}")   
#     # for q in queries:
#     #     try:
#     #         result = query_pipeline(query=q)
#     #         logger.info("Query: %s", q)
#     #         logger.info("Intent: %s", result.get("intent"))
#     #         logger.info("Entities: %s", result.get("entities"))
#     #         logger.info("Tools: %s", result.get("selected_tools"))
#     #         logger.info("Plan: %s", result.get("plan"))
#     #     except Exception as e:
#     #         logger.error("Error for query '%s': %s", q, e)
    

import os
import logging
import time
import json
from typing import Any, Dict, List, Optional

import dspy
import google.generativeai as genai
from dspy import Signature, InputField, OutputField


# -----------------------------------------------------
# 1. Logging Configuration
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------
# 2. GeminiLM CLASS
# -----------------------------------------------------
class GeminiLM(dspy.LM):
    """DSPy-compatible LLM wrapper for Google Gemini models."""

    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def _retry_request(self, func, *args, retries: int = 3, delay: int = 2, **kwargs) -> Any:
        """Retry mechanism for Gemini API calls."""
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning("Gemini API call failed (attempt %d/%d): %s", attempt + 1, retries, e)
                if attempt < retries - 1:
                    time.sleep(delay)
        raise RuntimeError("Gemini API call failed after multiple retries.")

    def __call__(self, *args, **kwargs) -> List[str]:
        """Handle DSPy prompt/message/query inputs."""
        prompt: Optional[str] = kwargs.get("prompt")
        messages: Optional[List[Dict[str, Any]]] = kwargs.get("messages")
        query: Optional[str] = kwargs.get("query")

        if prompt:
            response = self._retry_request(self._model.generate_content, prompt)
            return [response.text]

        elif messages:
            gemini_messages = []
            for m in messages:
                role = "user" if m.get("role") == "system" else m.get("role", "user")
                content = m.get("content", "")
                gemini_messages.append({"role": role, "parts": [{"text": content}]})
            response = self._retry_request(self._model.generate_content, gemini_messages)
            return [response.text]

        elif query:
            response = self._retry_request(self._model.generate_content, query)
            return [response.text]

        else:
            raise ValueError("No valid input provided to GeminiLM")


# -----------------------------------------------------
# 3. CONFIGURE DSPy with Gemini
# -----------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("Missing environment variable: GEMINI_API_KEY")

logger.info("Configuring DSPy with GeminiLM...")
dspy.configure(lm=GeminiLM("gemini-2.5-flash", GEMINI_API_KEY))
logger.info("DSPy configured successfully.")


# -----------------------------------------------------
# 4. SIGNATURES
# -----------------------------------------------------
class QueryUnderstanding(Signature):
    """Extracts intent and entities from user query. Use canonical keys: 
    - user
    - project
    - goal
    - date
    """
    query = InputField(desc="The natural language query from the user.")
    intent = OutputField(desc="High-level intent: FindPendingFollowUps, GenerateProgressReport, GetCodeCoverageProgress.")
    entities = OutputField(desc="JSON with keys like user, project, goal, date.")


class ToolSelection(Signature):
    """Maps intents to tools."""
    intent = InputField()
    selected_tools = OutputField(desc="List of relevant tools to use from AVAILABLE_TOOLS.")


class PlanGeneration(Signature):
    """Generates execution plan based on query and tools."""
    query = InputField()
    selected_tools = InputField()
    plan = OutputField(desc="Step-by-step JSON plan with tool calls and processing instructions.")


# -----------------------------------------------------
# 5. AVAILABLE TOOLS
# -----------------------------------------------------
AVAILABLE_TOOLS: List[str] = [
    "get_user_identities",
    "list_open_issues",
    "list_unread_emails",
    "list_recent_chats",
    "get_project_progress",
    "get_code_coverage_goal_progress"
]


# -----------------------------------------------------
# 6. ENTITY + INTENT NORMALIZATION
# -----------------------------------------------------
def normalize_entities(entities: Any) -> Dict[str, Any]:
    """Normalize LLM entity outputs to consistent schema."""
    if isinstance(entities, str):
        try:
            entities = json.loads(entities)
        except Exception:
            entities = {}
    if not isinstance(entities, dict):
        entities = {}

    normalized = {}
    for k, v in entities.items():
        key = k.lower().strip()
        if key in ["project_name", "proj", "project"]:
            normalized["project"] = v
        elif key in ["user", "username", "assignee", "person"]:
            normalized["user"] = v
        elif key in ["goal", "goal_type", "metric"]:
            normalized["goal"] = v
        elif key in ["date", "time_frame", "period"]:
            normalized["date"] = v
        else:
            normalized[key] = v
    return normalized


def normalize_intent(intent: str) -> str:
    """Map LLM-generated intent to canonical intent name."""
    mapping = {
        "FindPendingFollowUps": "FindPendingFollowUps",
        "GenerateProgressReport": "GenerateProgressReport",
        "GetProgressReport": "GetCodeCoverageProgress",   # unify naming
        "GetCodeCoverageProgress": "GetCodeCoverageProgress",
    }
    return mapping.get(intent, intent)


# -----------------------------------------------------
# 7. TOOL SELECTION MODULE
# -----------------------------------------------------
class ToolSelectionModule(dspy.Module):
    """Selects tools for an intent based on config mapping."""

    def __init__(self, available_tools: List[str], intent_tool_map: Dict[str, List[str]]):
        super().__init__()
        self.available_tools = available_tools
        self.intent_tool_map = intent_tool_map
        self.predict = dspy.Predict(ToolSelection)

    def forward(self, intent: str) -> dspy.Prediction:
        selected = self.intent_tool_map.get(intent, [])
        selected = [t for t in selected if t in self.available_tools]
        return dspy.Prediction(selected_tools=selected)


# -----------------------------------------------------
# 8. PLAN GENERATION MODULE
# -----------------------------------------------------
class PlanGenerationModule(dspy.Module):
    """Generates execution plan with normalized entities."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(PlanGeneration)

    def forward(self, query: str, selected_tools: List[str], entities: Any, intent: Optional[str] = None) -> dspy.Prediction:
        entities = normalize_entities(entities)
        intent = normalize_intent(intent or "")

        plan = []

        if intent == "FindPendingFollowUps":
            plan = [
                {"step": 1, "description": "Find user identities", "tool_call": f"get_user_identities('{entities.get('user', 'me')}')"},
                {"step": 2, "description": "Fetch open issues", "tool_call": "list_open_issues(tool_id, user_identity)"},
                {"step": 3, "description": "Fetch unread emails", "tool_call": "list_unread_emails(user_identity)"},
                {"step": 4, "description": "Fetch recent chats", "tool_call": f"list_recent_chats(tool_id, user_identity, '{entities.get('date', 'today')}')"},
                {"step": 5, "description": "Consolidate", "processing": "deduplicate_and_consolidate"},
                {"step": 6, "description": "Format JSON", "output_format": "json_with_source_links"}
            ]

        elif intent == "GenerateProgressReport":
            plan = [
                {"step": 1, "description": "Get project progress", "tool_call": f"get_project_progress('{entities.get('project', 'unknown')}')"},
                {"step": 2, "description": "Format report", "processing": "format_project_report"},
                {"step": 3, "description": "Generate document", "output_format": "document_report"}
            ]

        elif intent == "GetCodeCoverageProgress":
            plan = [
                {"step": 1, "description": "Retrieve code coverage", "tool_call": "get_code_coverage_goal_progress()"},
                {"step": 2, "description": "Summarize", "output_format": "simple_text_summary"}
            ]

        if plan:
            return dspy.Prediction(plan=plan)

        # fallback to LLM if unknown intent
        return self.predict(query=query, selected_tools=selected_tools)


# -----------------------------------------------------
# 9. QUERY PIPELINE
# -----------------------------------------------------
class QueryPipeline(dspy.Module):
    def __init__(self, intent_tool_map: Dict[str, List[str]]):
        super().__init__()
        self.understand = dspy.Predict(QueryUnderstanding)
        self.select_tools = ToolSelectionModule(AVAILABLE_TOOLS, intent_tool_map)
        self.generate_plan = PlanGenerationModule()

    def forward(self, query: str) -> Dict[str, Any]:
        understood = self.understand(query=query)
        intent = normalize_intent(understood.intent)
        entities = normalize_entities(understood.entities)
        tools = self.select_tools(intent=intent)
        plan = self.generate_plan(query=query, selected_tools=tools.selected_tools, entities=entities, intent=intent)
        return {
            "query": query,
            "intent": intent,
            "entities": entities,
            "selected_tools": tools.selected_tools,
            "plan": plan.plan
        }


# -----------------------------------------------------
# 10. EXECUTION
# -----------------------------------------------------
if __name__ == "__main__":
    INTENT_TOOL_MAP = {
        "FindPendingFollowUps": ["get_user_identities", "list_open_issues", "list_unread_emails", "list_recent_chats"],
        "GenerateProgressReport": ["get_project_progress"],
        "GetCodeCoverageProgress": ["get_code_coverage_goal_progress"],
    }

    queries = [
        "Find my pending open follow ups",
        "Generate a report on the current progress on Project X.",
        "Find the progress on the current code-coverage goal for the company"
    ]

    query_pipeline = QueryPipeline(INTENT_TOOL_MAP)

    logger.info("Running pipeline...")
    # for q in queries:
    #     try:
    #         result = query_pipeline(query=q)
    #         print("\nQuery:", q)
    #         print("Intent:", result.get("intent"))
    #         print("Entities:", result.get("entities"))
    #         print("Tools:", result.get("selected_tools"))
    #         print("Plan:", result.get("plan"))
    #         print("="*50)
    #     except Exception as e:
    #         print(f"Error for query '{q}': {e}")

    for q in queries:
         try:
             result = query_pipeline(query=q)
             logger.info("Query: %s", q)
             logger.info("Intent: %s", result.get("intent"))
             logger.info("Entities: %s", result.get("entities"))
             logger.info("Tools: %s", result.get("selected_tools"))
             logger.info("Plan: %s", result.get("plan"))
         except Exception as e:
             logger.error("Error for query '%s': %s", q, e)