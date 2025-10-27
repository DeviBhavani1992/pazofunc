#!/bin/bash

# Variables
RESOURCE_GROUP="func-rg-proj"
VM_NAME="myDS2VM"
LOCATION="eastus"          # change if needed
VM_SIZE="Standard_DS2_v2"
ADMIN_USER="azureuser"
EXTRA_DISK_NAME="extraDisk"
EXTRA_DISK_SIZE=50          # in GB

echo "🚀 Creating VM $VM_NAME in resource group $RESOURCE_GROUP..."

# 1️⃣ Ensure resource group exists
az group create --name $RESOURCE_GROUP --location $LOCATION

# 2️⃣ Create VM
az vm create \
  --resource-group $RESOURCE_GROUP \
  --name $VM_NAME \
  --image UbuntuLTS \
  --size $VM_SIZE \
  --admin-username $ADMIN_USER \
  --generate-ssh-keys \
  --storage-sku Standard_LRS

# 3️⃣ Open SSH port
az vm open-port --port 22 --resource-group $RESOURCE_GROUP --name $VM_NAME

# 4️⃣ Create extra managed disk
az disk create \
  --resource-group $RESOURCE_GROUP \
  --name $EXTRA_DISK_NAME \
  --size-gb $EXTRA_DISK_SIZE \
  --sku Standard_LRS

# 5️⃣ Attach extra disk to VM
az vm disk attach \
  --resource-group $RESOURCE_GROUP \
  --vm-name $VM_NAME \
  --name $EXTRA_DISK_NAME

# 6️⃣ Show VM info
az vm show --resource-group $RESOURCE_GROUP --name $VM_NAME -d -o table

echo "✅ VM $VM_NAME created successfully!"

