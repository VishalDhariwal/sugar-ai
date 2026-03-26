# Copyright (C) 2024 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# Copyright (C) 2024 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License...

"""
Main entry point for Sugar-AI application.
"""

import uvicorn
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app import create_app
from app.database import get_db, create_tables
from app.auth import sync_env_keys_to_db
from app.config import settings

# Setup logging
logger = logging.getLogger("sugar-ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles the startup and shutdown lifecycle of the application.
    """
    try:
        db = next(get_db())
        sync_env_keys_to_db(db)
        create_tables()

        logger.info(f"Starting Sugar-AI with model: {settings.DEFAULT_MODEL}")
    except Exception as e:
        logger.error(f"Failed to initialize app during startup: {e}")
        raise e

    yield

    logger.info("Shutting down Sugar-AI...")


# Create FastAPI app with lifespan
app = create_app(lifespan=lifespan)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )