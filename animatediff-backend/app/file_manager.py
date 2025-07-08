import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import mimetypes
import asyncio
from .models import VideoResultResponse

logger = logging.getLogger(__name__)

class FileManager:
    def __init__(self, output_dir: str = "./outputs", max_stored_videos: int = 100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_stored_videos = max_stored_videos
        
        # Create subdirectories
        self.videos_dir = self.output_dir / "videos"
        self.thumbnails_dir = self.output_dir / "thumbnails"
        self.videos_dir.mkdir(exist_ok=True)
        self.thumbnails_dir.mkdir(exist_ok=True)
        
        logger.info(f"File manager initialized with output dir: {self.output_dir}")
    
    def get_video_path(self, job_id: str) -> Path:
        """Get the path for a video file"""
        return self.videos_dir / f"{job_id}.mp4"
    
    def get_thumbnail_path(self, job_id: str) -> Path:
        """Get the path for a thumbnail file"""
        return self.thumbnails_dir / f"{job_id}_thumb.jpg"
    
    async def save_video(self, job_id: str, video_data: bytes) -> str:
        """Save video data to file"""
        video_path = self.get_video_path(job_id)
        
        try:
            with open(video_path, 'wb') as f:
                f.write(video_data)
            
            logger.info(f"Video saved for job {job_id}: {video_path}")
            return str(video_path)
            
        except Exception as e:
            logger.error(f"Error saving video for job {job_id}: {e}")
            raise
    
    async def save_thumbnail(self, job_id: str, thumbnail_data: bytes) -> str:
        """Save thumbnail data to file"""
        thumbnail_path = self.get_thumbnail_path(job_id)
        
        try:
            with open(thumbnail_path, 'wb') as f:
                f.write(thumbnail_data)
            
            logger.info(f"Thumbnail saved for job {job_id}: {thumbnail_path}")
            return str(thumbnail_path)
            
        except Exception as e:
            logger.error(f"Error saving thumbnail for job {job_id}: {e}")
            raise
    
    def file_exists(self, file_path: str) -> bool:
        """Check if a file exists"""
        return Path(file_path).exists()
    
    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        try:
            return Path(file_path).stat().st_size
        except FileNotFoundError:
            return 0
    
    def get_file_mime_type(self, file_path: str) -> str:
        """Get MIME type of a file"""
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or 'application/octet-stream'
    
    async def delete_video_files(self, job_id: str):
        """Delete video and thumbnail files for a job"""
        video_path = self.get_video_path(job_id)
        thumbnail_path = self.get_thumbnail_path(job_id)
        
        for file_path in [video_path, thumbnail_path]:
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Deleted file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete file {file_path}: {e}")
    
    async def get_stored_videos(self) -> List[Dict[str, any]]:
        """Get list of all stored videos with metadata"""
        videos = []
        
        for video_file in self.videos_dir.glob("*.mp4"):
            try:
                job_id = video_file.stem
                thumbnail_path = self.get_thumbnail_path(job_id)
                
                stat = video_file.stat()
                video_info = {
                    "job_id": job_id,
                    "video_path": str(video_file),
                    "thumbnail_path": str(thumbnail_path) if thumbnail_path.exists() else None,
                    "size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime),
                }
                
                videos.append(video_info)
                
            except Exception as e:
                logger.warning(f"Error reading video file {video_file}: {e}")
        
        # Sort by creation time, newest first
        videos.sort(key=lambda x: x["created_at"], reverse=True)
        return videos
    
    async def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old video files"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        deleted_count = 0
        
        for video_file in self.videos_dir.glob("*.mp4"):
            try:
                stat = video_file.stat()
                if datetime.fromtimestamp(stat.st_ctime) < cutoff_time:
                    job_id = video_file.stem
                    await self.delete_video_files(job_id)
                    deleted_count += 1
                    
            except Exception as e:
                logger.warning(f"Error cleaning up file {video_file}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old video files")
        return deleted_count
    
    async def enforce_storage_limit(self):
        """Enforce maximum number of stored videos"""
        videos = await self.get_stored_videos()
        
        if len(videos) <= self.max_stored_videos:
            return
        
        # Delete oldest videos
        videos_to_delete = videos[self.max_stored_videos:]
        deleted_count = 0
        
        for video_info in videos_to_delete:
            try:
                await self.delete_video_files(video_info["job_id"])
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Error deleting video {video_info['job_id']}: {e}")
        
        logger.info(f"Deleted {deleted_count} videos to enforce storage limit")
    
    def get_storage_info(self) -> Dict[str, any]:
        """Get storage information"""
        try:
            total_size = 0
            video_count = 0
            
            for video_file in self.videos_dir.glob("*.mp4"):
                total_size += video_file.stat().st_size
                video_count += 1
            
            # Get disk usage
            statvfs = os.statvfs(self.output_dir)
            disk_total = statvfs.f_frsize * statvfs.f_blocks
            disk_free = statvfs.f_frsize * statvfs.f_bavail
            disk_used = disk_total - disk_free
            
            return {
                "video_count": video_count,
                "total_size": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "disk_total": disk_total,
                "disk_free": disk_free,
                "disk_used": disk_used,
                "disk_usage_percent": (disk_used / disk_total) * 100,
                "max_stored_videos": self.max_stored_videos
            }
            
        except Exception as e:
            logger.error(f"Error getting storage info: {e}")
            return {
                "video_count": 0,
                "total_size": 0,
                "total_size_mb": 0,
                "error": str(e)
            }

# Global file manager instance
file_manager = FileManager(
    output_dir=os.getenv("OUTPUT_DIR", "./outputs"),
    max_stored_videos=int(os.getenv("MAX_STORED_VIDEOS", "100"))
)