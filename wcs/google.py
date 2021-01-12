from io import StringIO, BytesIO
#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import requests
def google_drive_share(id:str, binaryfile:bool=False, return_as:str=None, stream:bool=True, textencoding:str='utf-8'):
    '''
    Provides a method to open Google Drive Share linked files with pandas.read_csv() 
    or pandas.read_excel(). Might work with others, not tested yet.  

    Args
    ----
    *  `id` *:str* The shareing id. Google share links look like 
       https://drive.google.com/file/d/1uaDAGFiRmNj80FimGnHH5hNENkvjCYHv/view?usp=sharing 
       having the id enclosed between `.../file/d/` and `/view...`. in the 
       example above, the id is `1uaDAGFiRmNj80FimGnHH5hNENkvjCYHv` 
    *  `binaryfile` *:bool (default False)* For read_csv the response must be a 
        text string-like object, use `binaryfile=False`  
        For pandas.read_excel() the response must be the http response undecoded, so use 
        `binaryfile=True`
        
    *   `return_as` *:str (default None)*   
                            for binaryfile=True:
           'raw'            return the binary "content" of the response unchanged. 
                            binaryfile must be True or None
           'bytearray'      return the "content" wrapped in a bytearray instance
           'bytesio'        return the "content" wrapped in a BytesIO stream wrapper,
                            Default for binaryfile=True
           'response'       return the "naked" response object unchanged

                            for binaryfile=False:
           'text'           return the text decoded response as string
           'stringio'       return the text wrapped in a StringIO Object for file-like operations.
                            Default for binaryfile=False

    *   `stream` *:bool (default True)* enable streaming of the response

    *  `textencoding` *:str (default='utf-8)* For text (binaryfile==False) answers 
        the encoder to be used to encode the bytestream to text.  
    
    Examples
    --------
    1. Excel File Sample  
       ```
       pd.read_excel(google_drive_share('1uaDAGFiRmNj80FimGnHH5hNENkvjCYHv', 
         binaryfile=True), skiprows=1, skipfooter=5, usecols='A:I')
       ```

    1. CSV File Sample (Foodfacts)  
       ```
       pd.read_csv(google_drive_share(id = '1L4tDDnZWwPDjeRKlleoYHgic2_xNhZ8i', 
         binaryfile=False), sep='\\t', header=0, low_memory=False, nrows=10)
       ```
    '''

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None    
    
    baseurl = 'https://drive.google.com'
    if '/' in id: # it looks like a full URL
        id = id.split('/')[-2]
    URL = baseurl + '/uc?export= ' + id
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = stream)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = stream)
    sr = response

    if binaryfile:
        if return_as == 'raw':
            return sr.content
        elif return_as == 'bytearray':
            return bytearray(sr.content)
        elif return_as =='response':
            return sr
        else:
        #if return_as is None or return_as=='bytesio'
            return BytesIO(sr.content) # v0.6.8 wrap in BytesIO to enable pandas.read_parquet and ZipFile, solves #4
    
    # text file encoding
    sr.encoding=textencoding
    if return_as == 'text':
        return sr.text
    else: # stringio or None
        return StringIO(sr.text)
