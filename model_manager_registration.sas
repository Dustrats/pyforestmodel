/* 1. Define filenames */
filename model_f "/tmp/sas_model_RF.astore" recfm=n;
filename webbody temp;
filename resp temp;

/* 2. Build the Body with CRLF (Critical for API Parsing) */
data _null_;
   file webbody recfm=v lrecl=32767 termstr=crlf;
   put "--boundary";
   /* Note: using 'contents' as the name is the standard for the MM API */
   put 'Content-Disposition: form-data; name="contents"; filename="Forest_RF.astore"';
   put "Content-Type: application/octet-stream";
   put ; 
run;

/* 3. Append the binary data */
data _null_;
   length file_in $1;
   infile model_f recfm=n;
   file webbody mod recfm=n;
   input file_in @@;
   put file_in @@;
run;

/* 4. Append the closing boundary */
data _null_;
   file webbody mod termstr=crlf;
   put ;
   put "--boundary--";
run;

/* ... Keep your Steps 1 through 4 exactly as they are ... */
/* Step 4 creates the macro variable &model_id */

/* --- INSERT THE NEW BLOCK HERE --- */
proc python;
submit;
import os
import requests

def upload_astore():
    # Attempt to get the current model_id from SAS
    try:
        model_id = SAS.get_value('model_id')
    except:
        model_id = "bd736855-894a-4233-9109-9d5124039932"
    
    url = f"https://viya-cauki.unx.sas.com/modelRepository/models/{model_id}/contents"
    file_path = "/tmp/sas_model_RF.astore"
    token = os.environ.get('SAS_CLIENT_TOKEN')
    
    headers = {'Authorization': f'Bearer {token}'}
    
    # We provide the file under 'file', 'contents', and 'model' 
    # to satisfy whatever specific naming convention your API version requires.
    with open(file_path, 'rb') as f:
        files = {
            'file': ('Forest_RF.astore', f, 'application/octet-stream'),
            'contents': ('Forest_RF.astore', f, 'application/octet-stream')
        }
        
        print(f"--- Final Attempt: Uploading to {model_id} ---")
        # Note: verify=False handles internal SSL certificates
        response = requests.post(url, headers=headers, files=files, verify=False)
    
    if response.status_code in [200, 201]:
        print("✅ SUCCESS: The .astore file is officially registered!")
    else:
        print(f"❌ Still failing: {response.status_code}")
        print(response.text)

upload_astore()
endsubmit;
quit;

/* 5. Execute the HTTP POST */
proc http 
   url="&viya_api_url./modelRepository/models/&model_id/contents"
   method="POST"
   oauth_bearer=sas_services
   ct="multipart/form-data; boundary=boundary"
   in=webbody
   out=resp;
run;

/* Final check of the response */
data _null_;
   infile resp;
   input;
   put _infile_;
run;