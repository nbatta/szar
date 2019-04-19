#!/usr/local/bin/bash

set -e

BACKUPDIR="/Users/dylan/Documents/szar-backups"
TIMESTAMP=`date +%Y-%m-%d`

tar cf - userdata input/pipeline.ini | gzip -9 > szar_untracked_data_backup.tar.gz

mv szar_untracked_data_backup.tar.gz "$BACKUPDIR/szar_untracked_data_backup_$TIMESTAMP.tar.gz"
