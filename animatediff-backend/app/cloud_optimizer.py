"""
Cloud optimization features for cost control and efficient resource usage
"""

import os
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import psutil

logger = logging.getLogger(__name__)

class CloudOptimizer:
    """Optimizes cloud resource usage and costs"""
    
    def __init__(self):
        self.is_cloud_run = os.getenv("K_SERVICE") is not None
        self.startup_time = time.time()
        self.last_activity = time.time()
        self.idle_shutdown_minutes = int(os.getenv("IDLE_SHUTDOWN_MINUTES", "30"))
        self.max_idle_time = self.idle_shutdown_minutes * 60
        
        logger.info(f"Cloud optimizer initialized (Cloud Run: {self.is_cloud_run})")
    
    def record_activity(self):
        """Record user activity to prevent idle shutdown"""
        self.last_activity = time.time()
    
    def get_uptime(self) -> float:
        """Get service uptime in seconds"""
        return time.time() - self.startup_time
    
    def get_idle_time(self) -> float:
        """Get time since last activity in seconds"""
        return time.time() - self.last_activity
    
    def should_shutdown(self) -> bool:
        """Check if service should shutdown due to inactivity"""
        if not self.is_cloud_run:
            return False
        
        idle_time = self.get_idle_time()
        return idle_time > self.max_idle_time
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage stats"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            disk_usage = psutil.disk_usage('/')
            
            stats = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_used_gb": disk_usage.used / (1024**3),
                "disk_total_gb": disk_usage.total / (1024**3),
                "disk_percent": (disk_usage.used / disk_usage.total) * 100,
                "uptime_hours": self.get_uptime() / 3600,
                "idle_minutes": self.get_idle_time() / 60
            }
            
            # Add GPU stats if available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_allocated = torch.cuda.memory_allocated(0)
                    stats.update({
                        "gpu_available": True,
                        "gpu_memory_total_gb": gpu_memory / (1024**3),
                        "gpu_memory_used_gb": gpu_allocated / (1024**3),
                        "gpu_memory_percent": (gpu_allocated / gpu_memory) * 100
                    })
                else:
                    stats["gpu_available"] = False
            except Exception as e:
                logger.warning(f"Could not get GPU stats: {e}")
                stats["gpu_available"] = False
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return {"error": str(e)}
    
    def estimate_cost(self, generation_time_seconds: float) -> Dict[str, float]:
        """Estimate cost for a video generation"""
        if not self.is_cloud_run:
            return {"estimated_cost_usd": 0.0, "note": "Local deployment"}
        
        # Cloud Run with GPU pricing (approximate)
        # NVIDIA L4: ~$2.50/hour, 4 vCPU: ~$0.24/hour, 16GB RAM: ~$0.032/hour
        gpu_cost_per_hour = 2.50
        cpu_cost_per_hour = 0.24
        memory_cost_per_hour = 0.032
        
        total_cost_per_hour = gpu_cost_per_hour + cpu_cost_per_hour + memory_cost_per_hour
        
        hours = generation_time_seconds / 3600
        estimated_cost = total_cost_per_hour * hours
        
        return {
            "estimated_cost_usd": round(estimated_cost, 4),
            "generation_time_seconds": generation_time_seconds,
            "hourly_rate_usd": total_cost_per_hour,
            "breakdown": {
                "gpu_cost": round(gpu_cost_per_hour * hours, 4),
                "cpu_cost": round(cpu_cost_per_hour * hours, 4),
                "memory_cost": round(memory_cost_per_hour * hours, 4)
            }
        }
    
    async def cleanup_old_files(self, max_age_minutes: int = 60):
        """Clean up old files to save storage costs"""
        try:
            output_dir = os.getenv("OUTPUT_DIR", "/tmp/outputs")
            if not os.path.exists(output_dir):
                return
            
            cutoff_time = time.time() - (max_age_minutes * 60)
            cleaned_count = 0
            
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                if os.path.isfile(file_path):
                    if os.path.getmtime(file_path) < cutoff_time:
                        try:
                            os.remove(file_path)
                            cleaned_count += 1
                        except Exception as e:
                            logger.warning(f"Could not remove {file_path}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old files")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization and cost stats"""
        stats = {
            "cloud_run": self.is_cloud_run,
            "uptime_minutes": self.get_uptime() / 60,
            "idle_minutes": self.get_idle_time() / 60,
            "will_shutdown_in_minutes": max(0, (self.max_idle_time - self.get_idle_time()) / 60),
            "should_shutdown": self.should_shutdown(),
            "idle_shutdown_threshold_minutes": self.idle_shutdown_minutes
        }
        
        # Add resource usage
        stats.update(self.get_resource_usage())
        
        return stats

# Global optimizer instance
cloud_optimizer = CloudOptimizer()