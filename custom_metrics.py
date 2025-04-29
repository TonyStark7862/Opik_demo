import time
from typing import List, Optional, Sequence
import pandas as pd

from evidently.base_metric import InputData, Metric, MetricResult
from evidently.metrics.base_metric import UsesRawDataMixin # To access raw series
from evidently.model.widget import BaseWidgetInfo
from evidently.renderers.base_renderer import MetricRenderer, MetricHtmlInfo


# --- Custom Metric 1: Response Time ---
class ResponseTimeResult(MetricResult):
    """Data class to store results for the response time metric."""
    column_name: str
    current_mean_time: Optional[float] = None
    current_median_time: Optional[float] = None
    reference_mean_time: Optional[float] = None
    reference_median_time: Optional[float] = None
    # You could add distribution plots, drift detection results etc. here

class ResponseTimeMetric(Metric[ResponseTimeResult]):
    """Calculates basic statistics for a response time column."""
    column_name: str # Name of the column containing response times (in seconds)

    def __init__(self, column_name: str, description: Optional[str] = None):
        self.column_name = column_name
        super().__init__(description=description)

    def calculate(self, data: InputData) -> ResponseTimeResult:
        """Calculates metrics based on input data."""
        current_mean = None
        current_median = None
        ref_mean = None
        ref_median = None

        if self.column_name not in data.current_data:
             raise ValueError(f"Column '{self.column_name}' not found in current data.")

        current_times = data.current_data[self.column_name].dropna()
        if not pd.api.types.is_numeric_dtype(current_times):
             raise ValueError(f"Column '{self.column_name}' in current data must be numeric.")
        if len(current_times) > 0:
            current_mean = current_times.mean()
            current_median = current_times.median()

        if data.reference_data is not None:
            if self.column_name not in data.reference_data:
                # Warning instead of error for reference data
                print(f"Warning: Column '{self.column_name}' not found in reference data.")
            else:
                ref_times = data.reference_data[self.column_name].dropna()
                if not pd.api.types.is_numeric_dtype(ref_times):
                     print(f"Warning: Column '{self.column_name}' in reference data is not numeric.")
                elif len(ref_times) > 0:
                    ref_mean = ref_times.mean()
                    ref_median = ref_times.median()

        return ResponseTimeResult(
            column_name=self.column_name,
            current_mean_time=current_mean,
            current_median_time=current_median,
            reference_mean_time=ref_mean,
            reference_median_time=ref_median,
        )

# --- Custom Metric 2: Keyword Presence ---
class KeywordPresenceResult(MetricResult):
    """Data class for keyword presence metric results."""
    column_name: str
    keywords: List[str]
    current_presence_count: int = 0
    current_doc_count: int = 0
    reference_presence_count: Optional[int] = None
    reference_doc_count: Optional[int] = None

    @property
    def current_presence_ratio(self) -> Optional[float]:
        if self.current_doc_count == 0:
            return None
        return self.current_presence_count / self.current_doc_count

    @property
    def reference_presence_ratio(self) -> Optional[float]:
        if self.reference_doc_count is None or self.reference_doc_count == 0:
            return None
        return (self.reference_presence_count or 0) / self.reference_doc_count


class KeywordPresenceMetric(Metric[KeywordPresenceResult], UsesRawDataMixin):
    """Checks for the presence of specific keywords in a text column."""
    column_name: str
    keywords: List[str]

    def __init__(self, column_name: str, keywords: List[str], description: Optional[str] = None):
        self.column_name = column_name
        # Ensure keywords are lowercase for case-insensitive matching
        self.keywords = [kw.lower() for kw in keywords]
        super().__init__(description=description)

    def calculate(self, data: InputData) -> KeywordPresenceResult:
        current_presence = 0
        current_docs = 0
        ref_presence = None
        ref_docs = None

        if self.column_name not in data.current_data:
             raise ValueError(f"Column '{self.column_name}' not found in current data.")

        current_series = self.get_current_column(data.current_data, data.column_mapping)
        current_docs = len(current_series)
        for text in current_series.astype(str).str.lower():
            if any(kw in text for kw in self.keywords):
                current_presence += 1

        if data.reference_data is not None:
            if self.column_name not in data.reference_data:
                 print(f"Warning: Column '{self.column_name}' not found in reference data.")
            else:
                ref_series = self.get_reference_column(data.reference_data, data.column_mapping)
                ref_docs = len(ref_series)
                ref_presence = 0
                for text in ref_series.astype(str).str.lower():
                    if any(kw in text for kw in self.keywords):
                        ref_presence += 1

        return KeywordPresenceResult(
            column_name=self.column_name,
            keywords=self.keywords,
            current_presence_count=current_presence,
            current_doc_count=current_docs,
            reference_presence_count=ref_presence,
            reference_doc_count=ref_docs,
        )

    # Helper to get the correct column using mapping or name
    def _get_column(self, dataset: pd.DataFrame, column_mapping: PipelineColumnMapping) -> pd.Series:
         if column_mapping.is_set(self.column_name):
             return dataset[column_mapping.get(self.column_name)]
         return dataset[self.column_name]

    def get_current_column(self, dataset: pd.DataFrame, column_mapping: PipelineColumnMapping) -> pd.Series:
         return self._get_column(dataset, column_mapping)

    def get_reference_column(self, dataset: pd.DataFrame, column_mapping: PipelineColumnMapping) -> pd.Series:
         return self._get_column(dataset, column_mapping)


# --- Renderers for Custom Metrics (Optional but good for UI) ---
# These tell Evidently how to display the custom metric results in reports.

class ResponseTimeRenderer(MetricRenderer):
    def render_html(self, obj: ResponseTimeMetric) -> List[MetricHtmlInfo]:
        result = obj.get_result()
        current_stats = f"Mean: {result.current_mean_time:.2f}s, Median: {result.current_median_time:.2f}s" if result.current_mean_time is not None else "N/A"
        ref_stats = f"Mean: {result.reference_mean_time:.2f}s, Median: {result.reference_median_time:.2f}s" if result.reference_mean_time is not None else "N/A"

        # Simple counter widget showing current and reference values
        widget = BaseWidgetInfo(
            title=f"Response Time ({result.column_name})",
            type="counter",
            size=2,
            params={
                "counters": [
                    {"value": current_stats, "label": "Current"},
                    {"value": ref_stats, "label": "Reference"},
                ]
            },
        )
        return [MetricHtmlInfo(f"response_time_{result.column_name}", widget, header=obj.description or f"Response Time: {result.column_name}")]

class KeywordPresenceRenderer(MetricRenderer):
     def render_html(self, obj: KeywordPresenceMetric) -> List[MetricHtmlInfo]:
        result = obj.get_result()
        curr_ratio = result.current_presence_ratio
        ref_ratio = result.reference_presence_ratio

        current_stats = f"{result.current_presence_count}/{result.current_doc_count} ({curr_ratio*100:.1f}%)" if curr_ratio is not None else f"{result.current_presence_count}/{result.current_doc_count}"
        ref_stats = f"{result.reference_presence_count}/{result.reference_doc_count} ({ref_ratio*100:.1f}%)" if ref_ratio is not None else f"{result.reference_presence_count or 'N/A'}/{result.reference_doc_count or 'N/A'}"


        widget = BaseWidgetInfo(
            title=f"Keyword Presence ({result.column_name})",
            type="counter",
            size=2,
            params={
                "counters": [
                    {"value": current_stats, "label": "Current"},
                    {"value": ref_stats, "label": "Reference"},
                ]
            },
            details=f"Keywords: {', '.join(result.keywords)}"
        )
        return [MetricHtmlInfo(f"keyword_presence_{result.column_name}", widget, header=obj.description or f"Keyword Presence: {result.column_name}")]


# Register renderers so Evidently knows how to display these metrics
evidently.metric_preset.register_metric(ResponseTimeMetric, ResponseTimeRenderer)
evidently.metric_preset.register_metric(KeywordPresenceMetric, KeywordPresenceRenderer)
