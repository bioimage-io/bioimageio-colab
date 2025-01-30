#!/bin/sh

if [ -f /app/service_id.txt ]; then
    SERVICE_INFO=$(cat /app/service_id.txt)
    WORKSPACE_ID=$(echo "$SERVICE_INFO" | cut -d'/' -f1)
    CLIENT_ID=$(echo "$SERVICE_INFO" | cut -d'/' -f2 | cut -d':' -f1)
    SERVICE_ID=$(echo "$SERVICE_INFO" | cut -d':' -f2)

    curl -sf "https://hypha.aicell.io/$WORKSPACE_ID/services/$CLIENT_ID:$SERVICE_ID/deployment_status?assert_status=True" || exit 1
else
    exit 1
fi
