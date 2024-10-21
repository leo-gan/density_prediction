import asyncio
import aiofiles
import os

async def stream_file(file_path: str, chunk_size: int = 1024 * 1024):
    """ Stream large files asynchronously """
    async with aiofiles.open(file_path, mode='rb') as f:
        while True:
            chunk = await f.read(chunk_size)
