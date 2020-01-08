import boto3

def glacier_restore(bucket_name, remote_folder, keyword='', days=60, tier='Standard'):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    obj_list = []

    for object in bucket.objects.all():
        folder, filename = ('/'.join(object.key.split("/")
                                     [:-1]), object.key.split("/")[-1])
        # only look in remote_folder and if any keyword(s) math filename
        if folder in remote_folder and any(k in filename for k in ([keyword]*isinstance(keyword, str) or keyword)):
            obj_list.append(s3.Object(object.bucket_name, object.key))

    if '' in keyword:
        press = input(
            f'Warning!: Do you really want to restore ALL files in {bucket.name}/{remote_folder}? (Y/N): ')
        if press != 'Y':
            print('Canceling restore')
            return
    restore_requestDict = {
        'Days': days,  # Number of days object is kept in S3 before it is removed
        'GlacierJobParameters': {
            'Tier': tier  # options: 'Expedited', 'Standard', 'Bulk'
        },
    }
    for object in obj_list:
        # if it's in Glacier:
        if object.storage_class == "GLACIER":
            # and not currently being restored:
            if object.restore is None:
                print("Requesting restore for --> " + object.key)
                object.restore_object(Bucket=object.bucket_name,
                                      Key=object.key,
                                      RestoreRequest=restore_requestDict)
            # list ones currently queued for restore
            elif 'ongoing-request="true"' in object.restore:
                print("Already in restore queue --> " + object.key)
            # list ones already restored
            elif 'ongoing-request="false"' in object.restore:
                print("Already restored --> "+object.key)
    print('---REQUEST DONE---')



def copy_files(source_bucket, dest_bucket, source_folder, dest_folder, keyword='', **kwargs):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(source_bucket)
    force_copy = False
    obj_list = []
    dest_list = []
    if 'force' in kwargs:
        force_copy = kwargs['force']
    press = input(
            f'Copying files from  {source_bucket} --> {dest_bucket}. Continue? (Y/N): ')
    if press != 'Y':
        print('Canceling copy')
        return

    for object in bucket.objects.all():
        folder, filename = ('/'.join(object.key.split("/")
                                     [:-1]), object.key.split("/")[-1])
        # only look in remote_folder and if any keyword(s) math filename
        if folder in source_folder and any(k in filename for k in ([keyword]*isinstance(keyword, str) or keyword)):
            obj_list.append(s3.Object(object.bucket_name, object.key))

    if force_copy == False:
        copy_bucket = s3.Bucket(dest_bucket)

        for object in copy_bucket.objects.all():
            copy_folder, copy_filename = ('/'.join(object.key.split("/")
                                         [:-1]), object.key.split("/")[-1])
            if copy_folder in dest_folder and any(k in copy_filename for k in ([keyword]*isinstance(keyword, str) or keyword)):
                dest_list.append(object.key.split("/")[-1])

    if '' in keyword:
        press = input(
            f'Warning!: Do you really want to copy ALL files in {bucket.name}/{source_folder}? (Y/N): ')
        if press != 'Y':
            print('Canceling copy')
            return

    for object in obj_list:
        copy_source = {
            'Bucket': object.bucket_name,
            'Key': object.key
         }
        filename = object.key.split("/")[-1]
        if filename in dest_list and force_copy == False:
            press = input(
                f'{object.key} already exists: Overwrite? (Y/N): ')
            if press != 'Y':
                print('Skipping')
                continue

        s3.meta.client.copy(copy_source, dest_bucket, dest_folder+f"/{filename}")
        print(f"Copied file : \n{source_bucket}/{object.key} --> {dest_bucket}/{dest_folder}/{filename}")
    print('---COPY DONE---')

def load_files(bucket_name, remote_folder, keyword=''):
    pass

