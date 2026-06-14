import os
import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout

# 1. Connect with high timeout (restoring 1M vectors takes time!)
client = weaviate.connect_to_local(
    additional_config=AdditionalConfig(
        timeout=Timeout(init=60, query=600, insert=3600) 
    )
)

BACKUP_ID = "research-papers-database" 

try:
    print(f"Starting restoration of backup: {BACKUP_ID}...")
    
    result = client.backup.restore(
        backup_id=BACKUP_ID,
        backend="filesystem",
        wait_for_completion=True # Script will wait until it's finished
    )

    if result.status == "SUCCESS":
        print("\n[SUCCESS] Database restored successfully!")
    else:
        print(f"\n[ERROR] Restoration failed: {result.error}")

except Exception as e:
    print(f"\n[CRITICAL] An error occurred: {e}")

finally:
    client.close()
