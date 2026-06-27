from datetime import datetime
from typing import List, Optional, Any, Dict
import json

from .base import FailureAdapter, RunEvent, FailedStepInfo

class KubernetesAdapter(FailureAdapter):
    @property
    def source_name(self) -> str:
        return "kubernetes"

    @property
    def available_signals(self) -> List[str]:
        # K8s failures often lack a commit_message/branch unless stamped on the pod
        # We start with base signals, and dynamic ones will be added in parse()
        return [
            "error_text",
            "detection_mode",
            "k8s_events"
        ]

    def parse(self, payload: Dict[str, Any]) -> RunEvent:
        """
        Parses a Kubernetes failure payload.
        Expected payload dict should look like:
        {
            "namespace": "default",
            "pod_name": "my-app-12345",
            "phase": "Failed",
            "reason": "CrashLoopBackOff",
            "logs": "Traceback ...",
            "events": ["Warning: Back-off restarting failed container"],
            "labels": {"commit_sha": "abc1234"}
        }
        """
        namespace = payload.get("namespace", "default")
        pod_name = payload.get("pod_name", "unknown")
        repo = f"{namespace}/{pod_name}"
        
        reason = payload.get("reason", "Error")
        logs = payload.get("logs", "")
        events = payload.get("events", [])
        labels = payload.get("labels", {})

        # Attempt to extract Git metadata if present in labels/annotations
        commit_sha = labels.get("commit_sha") or labels.get("app.kubernetes.io/revision")
        
        # Combine logs and events for error text
        error_text_combined = f"REASON: {reason}\nLOGS:\n{logs}\nEVENTS:\n" + "\n".join(events)
        
        failed_step = FailedStepInfo(
            job_file=pod_name,
            runner_image=payload.get("image", "unknown"),
            step_index=0,
            step_type="container",
            step_label=f"Pod {pod_name} ({reason})",
            step_duration_sec=None,
            error_text=error_text_combined,
            detection_mode="per_step_error"
        )

        return RunEvent(
            source=self.source_name,
            run_id=f"k8s-{pod_name}-{datetime.utcnow().timestamp()}",
            repo=repo,
            workflow="kubernetes-deployment",
            event="pod_failure",
            commit_sha=commit_sha,
            conclusion="failure",
            started_at=datetime.utcnow().isoformat(),
            duration_sec=None,
            n_jobs=1,
            failed_jobs_count=1,
            failed_steps=[failed_step],
            failure_detection="per_step_error",
            has_log_insights=True,
            available_signals=self.available_signals,
            metadata={"k8s_events": events, "k8s_reason": reason}
        )

    def fetch_from_api(self, namespace: str, pod_name: str) -> RunEvent:
        """
        Actively queries the Kubernetes API for the given Pod.
        """
        try:
            from kubernetes import client, config
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
                
            v1 = client.CoreV1Api()
            pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
            
            # Fetch logs
            try:
                logs = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace, tail_lines=50)
            except Exception:
                logs = "Logs not available"
                
            # Fetch events
            events_list = v1.list_namespaced_event(namespace=namespace, field_selector=f"involvedObject.name={pod_name}")
            warning_events = [e.message for e in events_list.items if e.type == "Warning"]
            
            payload = {
                "namespace": namespace,
                "pod_name": pod_name,
                "phase": pod.status.phase,
                "reason": pod.status.container_statuses[0].state.waiting.reason if pod.status.container_statuses and pod.status.container_statuses[0].state.waiting else "Unknown",
                "logs": logs,
                "events": warning_events,
                "labels": pod.metadata.labels or {},
                "image": pod.spec.containers[0].image if pod.spec.containers else "unknown"
            }
            return self.parse(payload)
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch K8s data: {e}")
