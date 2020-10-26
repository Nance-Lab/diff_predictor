import boto3

# Note: All methods in this file require either to be in an AWS EC2 instance
#       or to have an AWSCLI installed and configured.

def get_s3_keys(bucket, prefix='', suffix=''):
    """
    Generate the keys in an S3 bucket. Uses bucket name with optional
    prefix or suffix values.

    Parameters
    ----------
    bucket : string :
        Name of the S3 bucket to look through.
    prefix : string :
        Only generate keys that start with this prefix. Includes folder name.
    suffix : string :
        Only generate keys that end with this suffix
        useful for specifying filetype).
    """
    
    
    s3 = boto3.client('s3')
    kwargs = {'Bucket': bucket, 'Prefix': prefix}
    while True:
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            key = obj['Key']
            if key.endswith(suffix):
                yield key
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break

            
def glacier_restore(bucket_name, remote_folder, keyword='', days=60, tier='Standard'):
    """"
    Restores files stores in aws s3 given a keyword or set of keywords. Will
    search through given bucket/folder for keywords and if the files are in
    glacier state, will restore for given days and given gacier job parameters.

    Parameters
    ----------
    bucket_name : string :
        Name of the S3 bucket to look through
    remote_folder : string :
        Name of the folder within the bucket to look through
    keyword : string or list of strings :
        Keyword(s) to look through all files, this can be any keyword within
        the file/object
    days : int :
        How many days to keep file out of glacier
    tier : string :
        glacier jobrestore paraeter. Options are 'Expedited', 'Standard', and
        'Bulk'.

        Expedited - 1-5 minutes
        Standard - 3-5 hours
        Bulk - 5-12 hours

        Note that Expedited option requires Provisioned Capacity before using. For more info see
        HHhttps://docs.aws.amazon.com/AmazonS3/latest/user-guide/restore-archived-objects.html

    """

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
        'Days': days,
        'GlacierJobParameters': {
            'Tier': tier
        },
    }
    for object in obj_list:
        if object.storage_class == "GLACIER":
            # check if objects that are queued fr restore
            if 'ongoing-request="true"' in object.restore:
                print("Already in restore queue --> " + object.key)
                continue
            # check if objects that are already restored
            elif 'ongoing-request="false"' in object.restore:
                print("Already restored --> "+object.key)
                continue
            # check if object is not in restore
            print("Requesting restore for --> " + object.key)
            object.restore_object(Bucket=object.bucket_name,
                                    Key=object.key,
                                    RestoreRequest=restore_requestDict)
    print('---REQUEST DONE---')


def copy_files(source_bucket, dest_bucket, source_folder, dest_folder, keyword='', **kwargs):

    """
    Copies files in aws s3 given a keyword or set of keywords. Will search through
    given bucket/folder for keywords and will copy files that contain the matching
    keyword to the destination bucket/folder.

    Parameters
    ----------
    source_bucket : string :
        Name of the S3 bucket to look through
    dest_bucket : string :
        Name of the S3 bucket to copy to
    source_folder : string :
        Name of the folder within the bucket to look through
    dest_folder : string :
        Name of the folder within the bucket to copy to
    keyword : string or list of strings :
        Keyword(s) to look through all files, this can be any keyword within
        the file/object
    """


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

