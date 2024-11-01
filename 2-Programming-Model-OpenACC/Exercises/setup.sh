#!/usr/bin/env bash

echo "Resetting environment to global course environment"

module purge
source $JSCCOURSE_DIR_GROUP/common/environment/modules.sh