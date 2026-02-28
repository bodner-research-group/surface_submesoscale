#!/bin/bash

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

REMOTE="y_si@eofe7.mit.edu:/orcd/data/abodner/002/ysi/surface_submesoscale/Manuscript_Data/"
LOCAL="/Users/ysi/surface_submesoscale/Manuscript_Data/"

rsync -av --update "$REMOTE"* "$LOCAL"

ssh-agent -k

