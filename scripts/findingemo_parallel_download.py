#!/usr/bin/env python3

import os
import json
import time
import asyncio
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from pathlib import Path
import argparse

class ParallelImageDownloader:
    def __init__(self, target_dir="data", max_workers=20, timeout=30):
        self.target_dir = Path(target_dir)
        self.max_workers = max_workers
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = None
        self.downloaded_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def load_image_data(self):
        """Load the JSON file with image URLs and paths"""
        # Find the JSON file in the installed package
        try:
            import findingemo_light
            package_dir = Path(findingemo_light.__file__).parent
            json_file = package_dir / "data" / "dataset_urls_exploded.json"
            
            with open(json_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading image data: {e}")
            return []
    
    def group_by_image_path(self, json_data):
        """Group URLs by image path to handle multiple URL fallbacks"""
        grouped = {}
        for item in json_data:
            rel_path = item['rel_path']
            if rel_path not in grouped:
                grouped[rel_path] = []
            grouped[rel_path].append(item)
        
        # Sort by idx_url to try URLs in order
        for rel_path in grouped:
            grouped[rel_path].sort(key=lambda x: x['idx_url'])
        
        return grouped
    
    async def download_image(self, url, file_path):
        """Download a single image from URL"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    # Create directory if it doesn't exist
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write image data
                    async with aiofiles.open(file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    return True
                else:
                    return False
        except Exception:
            return False
    
    async def try_wayback_machine(self, original_url, file_path):
        """Try downloading from Wayback Machine as fallback"""
        try:
            # Query Wayback Machine API
            wayback_api = "https://archive.org/wayback/available"
            payload = {"url": original_url}
            
            async with self.session.post(wayback_api, data=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    archived_snapshots = result.get('results', [{}])[0].get('archived_snapshots', {})
                    
                    if archived_snapshots:
                        archived_url = archived_snapshots['closest']['url']
                        # Convert to direct image URL
                        idx = archived_url.find(original_url)
                        if idx < 0 and original_url.startswith('http://'):
                            idx = archived_url.find(original_url.replace('http://', 'https://'))
                        
                        if idx >= 0:
                            archived_url = archived_url[:idx - 1] + "if_" + archived_url[idx - 1:]
                            return await self.download_image(archived_url, file_path)
            return False
        except Exception:
            return False
    
    async def process_image(self, rel_path, url_data_list):
        """Process a single image with multiple URL fallbacks"""
        file_path = self.target_dir / rel_path.lstrip('/')
        
        # Skip if already exists
        if file_path.exists():
            self.skipped_count += 1
            print(f"✓ Skipped (exists): {rel_path}")
            return True
        
        # Try each URL in order
        for url_data in url_data_list:
            url = url_data['url']
            idx_url = url_data['idx_url']
            
            print(f"🔄 Trying URL {idx_url} for {rel_path}")
            
            # Try direct download
            if await self.download_image(url, file_path):
                self.downloaded_count += 1
                print(f"✅ Downloaded: {rel_path}")
                return True
            
            # Try Wayback Machine
            if await self.try_wayback_machine(url, file_path):
                self.downloaded_count += 1
                print(f"✅ Downloaded via Wayback: {rel_path}")
                return True
        
        # All URLs failed
        self.failed_count += 1
        print(f"❌ Failed: {rel_path}")
        return False
    
    async def download_all(self):
        """Download all images with parallel processing"""
        print("Loading image data...")
        json_data = self.load_image_data()
        if not json_data:
            print("No image data found!")
            return
        
        print(f"Found {len(json_data)} URL entries")
        
        # Group by image path
        grouped_data = self.group_by_image_path(json_data)
        total_images = len(grouped_data)
        
        print(f"Processing {total_images} unique images with {self.max_workers} parallel workers...")
        
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def limited_process(rel_path, url_list):
            async with semaphore:
                return await self.process_image(rel_path, url_list)
        
        # Start all downloads
        start_time = time.time()
        tasks = [limited_process(rel_path, url_list) 
                for rel_path, url_list in grouped_data.items()]
        
        # Process all images
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"\n🎉 Download Summary:")
        print(f"   ✅ Downloaded: {self.downloaded_count}")
        print(f"   ⏭️  Skipped: {self.skipped_count}")
        print(f"   ❌ Failed: {self.failed_count}")
        print(f"   ⏱️  Time: {elapsed_time:.1f}s")
        print(f"   🚀 Speed: {(self.downloaded_count + self.skipped_count) / elapsed_time:.1f} images/sec")

async def main():
    parser = argparse.ArgumentParser(description='Parallel download of FindingEmo dataset')
    parser.add_argument('--target-dir', default='data', help='Target directory for downloads')
    parser.add_argument('--workers', type=int, default=20, help='Number of parallel workers')
    parser.add_argument('--timeout', type=int, default=30, help='Download timeout in seconds')
    
    args = parser.parse_args()
    
    async with ParallelImageDownloader(
        target_dir=args.target_dir, 
        max_workers=args.workers, 
        timeout=args.timeout
    ) as downloader:
        await downloader.download_all()

if __name__ == "__main__":
    asyncio.run(main())
