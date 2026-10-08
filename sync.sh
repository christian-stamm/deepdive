#!/bin/bash
zip -r tmp.zip . -x "*.git*" -x "*.docker*" -x "sync.zip" -x "*__pycache__*" -x "./res/*"