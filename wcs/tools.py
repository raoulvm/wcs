#import numpy as np
class HTMLtable:
    """
    HTML Table helper class. 

    """
    CELLTYPE = 'type'
    CELLCONTENTS = 'contents'
    CELLSPANDOWN = 'spandown'
    CELLSPANRIGHT = 'spanright'
    CELLTYPE_NORMAL         = 0
    CELLTYPE_BEGINSPANRIGHT = 1
    CELLTYPE_BEGINSPANDOWN  = 2
    CELLTYPE_SPANNED        = 4 
    def __init__(self, rows:int=2, cols:int=2, caption:str=None):
        if not (isinstance(rows, int) and isinstance(cols, int)):
            raise TypeError('rows and cols must be integers')
        if (rows<1) or (cols<1):
            raise ValueError('(rows<1) or (cols<1)')
        self.__rows=rows
        self.__cols=cols
        self.__caption=caption
        self.__cells = {row:{col:{HTMLtable.CELLTYPE:HTMLtable.CELLTYPE_NORMAL, 
                                  HTMLtable.CELLCONTENTS:'', 
                                  HTMLtable.CELLSPANRIGHT: 0, 
                                  HTMLtable.CELLSPANDOWN:0} for col in range(cols)} for row in range(rows)}
        return None
    def add_rows(self, nrows:int):
        if not isinstance(nrows, int):
            raise TypeError(f'excpeted int, got {type(nrows).__repr__(nrows)} ')
        if nrows<=0:
            raise ValueError(f' number of rows must not be smaller than 1')
        self.__cells.update( {row:{col:{HTMLtable.CELLTYPE:HTMLtable.CELLTYPE_NORMAL, 
                                HTMLtable.CELLCONTENTS:'', 
                                HTMLtable.CELLSPANRIGHT: 0, 
                                HTMLtable.CELLSPANDOWN:0} for col in range(self.__cols)} for row in range(self.__rows, self.__rows+nrows)} )
        self.__rows += nrows
        return self
    def add_columns(self, ncols:int):
        if not isinstance(ncols, int):
            raise TypeError(f'excpeted int, got {type(ncols).__repr__(ncols)} ')
        if ncols<=0:
            raise ValueError(f' number of ncols must not be smaller than 1')
        for row in range(self.__rows):
            self.__cells[row].update({col:{HTMLtable.CELLTYPE:HTMLtable.CELLTYPE_NORMAL, 
                                        HTMLtable.CELLCONTENTS:'', 
                                        HTMLtable.CELLSPANRIGHT: 0, 
                                        HTMLtable.CELLSPANDOWN:0} for col in range(self.__cols, self.__cols+ncols)})
        self.__cols += ncols
        return self
    def __getcell(self, row, col, attr:str=None):
        if attr is None:
            return self.__cells[row][col]
        else:
            return self.__cells[row][col][attr]
    @property
    def caption(self):
        return self.__caption

    @caption.setter
    def caption(self, caption:str):
        """
        Set the extra caption under the table.
        Use None to remove it.
        """
        self.__caption=caption
        return self

    def merge_cells(self, row_start, col_start, row_end=None, col_end=None):
        """
        Merge the cells into one span. Will fail if cells are already part of a span
        Enter first and last column, 0-indexed
        """
        if (row_end is None and col_end is None): 
            raise ValueError('merge must be at least 2 rows or two columns in size')
        if row_end is None: row_end = row_start
        if col_end is None: col_end = col_start
        if col_end-col_start<1 and row_end-row_start<1:
            raise ValueError(f'merge must be at least 2 rows or 2 columns in size. Given:  {row_start}, {col_start} to {row_end}, {col_end}')
        if col_end<col_start or row_end<row_start:
            raise ValueError(f'merge cannot start before end: Given {row_start}, {col_start} to {row_end}, {col_end}')            
        #check, all cells must be CELLTYPE==0
        if sum([self.__getcell(row, col, HTMLtable.CELLTYPE) for row in range(row_start, row_end+1) for col in range(col_start, col_end+1) ])>0:
            raise ValueError(f'Cells in the range {row_start}, {col_start}, {row_end}, {col_end} are already spanned')
        # set all cells to spanned
        for row in range(row_start, row_end+1):
            for col in range(col_start, col_end+1):
                self.__cells[row][col][HTMLtable.CELLTYPE] = HTMLtable.CELLTYPE_SPANNED
        # new type of left top cell
        ntLT = HTMLtable.CELLTYPE_NORMAL + (HTMLtable.CELLTYPE_BEGINSPANDOWN if row_end>row_start else 0) + (HTMLtable.CELLTYPE_BEGINSPANRIGHT if col_end>col_start else 0) 
        self.__cells[row_start][col_start][HTMLtable.CELLTYPE] = ntLT
        if (col_end-col_start)!=0:
            self.__cells[row_start][col_start][HTMLtable.CELLSPANRIGHT] = (col_end-col_start)+1
        if (row_end-row_start)!=0:
            self.__cells[row_start][col_start][HTMLtable.CELLSPANDOWN] = (row_end-row_start)+1
    
        return self

    def __setitem__(self, key, value):
        """
        write value(s) to the table.
        
        """
        # [row] int
        # [row,:] tuple(int,  slice(None, None, None))
        # [row, col] tuple(int, int)
        # [:, col] tuple( slice(None, None, None), int)
        # [row:row2, col] tuple(slice(row, row2, None), int)

        if isinstance(key, tuple) and len(key)==2 and isinstance(key[0], int) and isinstance(key[1], int):
            # row, col
            row, col = key
            if row>=self.__rows or col>=self.__cols:
                raise KeyError(f'{row}, {col} out of bounds {self.__rows}, {self.__cols} ')
            if self.__getcell(row, col, HTMLtable.CELLTYPE) == HTMLtable.CELLTYPE_SPANNED:
                raise KeyError(f'{row}, {col} part of spanned rows/columns ')
            self.__cells[row][col][HTMLtable.CELLCONTENTS] = str(value)
        else:
            raise NotImplementedError(f'key == {key} not yet supported')

    def _repr_html_(self)->str:
        head = """  <head> 
        <meta charset="utf-8">
        <style>
            table, th, td {border: 1px solid #AAAAAA; border-collapse: collapse; padding: 4px 8px}
        </style> 
        </head>
        <body><table>"""
        if self.__caption is not None:
            head += ' <caption>' + str(self.__caption) + '</caption> '
        tail = '</table>'
        bodystr = ''
        for row in range(0, self.__rows):
            bodystr += '<tr>'
            for col in range(0, self.__cols):
                if self.__getcell(row, col, HTMLtable.CELLTYPE)!=HTMLtable.CELLTYPE_SPANNED:
                    colspan = ''
                    rowspan = ''
                    if self.__getcell(row, col, HTMLtable.CELLTYPE) & HTMLtable.CELLTYPE_BEGINSPANRIGHT:
                        colspan = f' colspan="{self.__getcell(row, col, HTMLtable.CELLSPANRIGHT)}" '
                    if self.__getcell(row, col, HTMLtable.CELLTYPE) & HTMLtable.CELLTYPE_BEGINSPANDOWN:
                        rowspan = f' rowspan="{self.__getcell(row, col, HTMLtable.CELLSPANDOWN)}" '
                    bodystr += f'<td {colspan} {rowspan} >' + self.__getcell(row, col, HTMLtable.CELLCONTENTS) + '</td>'
                else:
                    pass
            bodystr += '</tr>'
        return head + bodystr + tail
    def __repr__(self)->str:
        return str(self.__cells)


import matplotlib.pyplot as plt
from typing import Callable
import math
import numpy as np
def pltfct(fct:Callable[[float],float]=math.sin, xmin:float=-10.0, xmax:float=10.0, step:float=0.1, label:str='function', title:str='', figsize=(10,5)):
    '''
    # Easy Function Plotter
    `fct`:function or list of functions to be plotted
    `xmin`:float lower bound 
    `xmax`:float upper bound
    `step` step size for the plot
    `label`:str or list(str) Legend labels for the function plots
    `title`:str Chart title
    '''
    lines = []
    # create the range
    xr = np.linspace(xmin, xmax, round((xmax-xmin)/step))
    fig, ax = plt.subplots(figsize=figsize)
    ymax = None
    ymin = None
    i = 0
    if isinstance(fct, list):
        if isinstance(label, list):
            if len(label)<len(fct):
                label.extend(['function '+str(i) for i in range(len(label), len(fct))])
        else:
            label = [label+' '+str(i) for i in range(0, len(fct))]
            #print(label)
        for fcti in fct:
            if callable(fcti):
                y = [fcti(x) for x in xr]
                ax.plot(xr, y, label=label[i])
                if ymin==None:
                    ymin = np.min(y)
                else:
                    ymin = min(ymin, np.min(y))
                if ymax==None:
                    ymax = np.max(y)
                else:
                    ymax = max(ymax, np.max(y))
                i += 1
    else:
        if callable(fct):
            y = [fct(x) for x in xr]
            ax.plot(xr, y, label=label)
            ymin = np.min(y)
            ymax = np.max(y)
    #print(lines)
    ax.set_title(title)
    # set the axes cross to 0 if 0 is in the x range
    if (xmin<=0) & (xmax>=0):
        ax.spines['left'].set_position(('data',0))
    else:
        ax.spines['left'].set_position(('axes',0))
    if (ymin<=0) & (ymax>=0):
        ax.spines['bottom'].set_position(('data',0))
    else:
        ax.spines['bottom'].set_position(('axes',0))    

    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.legend(loc=1)
    plt.show()

#============================================================

# to talk API
import json
import requests
import pandas as pd
import urllib3
import datetime

#for handling images
from PIL import Image as pil_image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Dict

class Tmdb_api_connector:
    def __init__(self, 
                 api_key:str,
                 language:str='en-US',
                 include_adult:bool=False,
                 ):
        self._api_key = api_key
        self._language = language
        self._include_adult = include_adult
        self._config = self._get_dict_from_url(url='https://api.themoviedb.org/3/configuration', paramsdict={})

    def _get_dict_from_url(self, url:str, paramsdict:Dict[str,str]):
        """
        Low level API call to get the web services response and 
        raises
        ------
        requests.HTTPError in case something went wrong with the web API
        JSONDecodeError    in case the answer is not parseable
        """
        # proper URL encoding, and add api key and language for each call
        all_keys = {**paramsdict, 
            'api_key':self._api_key,
            'language':self._language,
        }
        r = requests.get(url + '?' + urllib3.request.urlencode(all_keys) )
        if r.ok:
            j = json.loads(r.text)
            r.close()
            return j
        raise requests.HTTPError(f"HTTP ERROR {str(r)}", response=r)

    def _get_allpages(self, url:str, paramsdict:Dict[str,str]):
        """
        Data from the movie database might be paginated (returned in chunks of e.g. 20 records)
        raises
        ------
        requests.HTTPError in case something went wrong
        """
        r1 = self._get_dict_from_url(url, paramsdict)
        r = [r1]
        #display(r)
        if 'total_pages' in r1:
            # print('more than one page')
            for next_page in range(2, r1['total_pages']+1):
                # print(f"load page {next_page} ")
                r.append(self._get_dict_from_url(url, {**paramsdict, 'page':next_page}))
        # print(len(r))
        # print([len(rx['results']) for rx in r])
        results = [entry for rx in r for entry in rx['results']  ]

        return results

    def movie_query(self, querystring:str)->pd.DataFrame:
        result = tmdb._get_allpages(url='https://api.themoviedb.org/3/search/movie', paramsdict={'query':querystring, 'include_adult':self._include_adult})
        return pd.DataFrame.from_records(result) 

    def get_genres_dict(self)->Dict[int,str]:
        try: 
            # used cached list if any is there
            return self._genres
        except AttributeError:
            r = self._get_dict_from_url(url='https://api.themoviedb.org/3/genre/movie/list', paramsdict={})
            r = r['genres']
            r = {x['id']:x['name'] for x in r}
            self._genres = r
        return r

    def get_poster(self, poster_path:str, size:str=None)->pil_image.Image:
        if size is None:
            size = self._config['images']['poster_sizes'][4]
        if isinstance(size, int):
            size = tmdb._config['images']['poster_sizes'][min(size,len(tmdb._config['images']['poster_sizes'])-1)]
        r = requests.get(tmdb._config['images']['secure_base_url']+size+poster_path)
        if r.ok:
            im = pil_image.open(BytesIO(r.content))
            im.load() # force loading so we can close the connection
            r.close()
            return im
        raise requests.HTTPError(f"HTTP ERROR {str(r)}", response=r)

    def get_full_movie_list(self):
        try:
            return self._full_list
        except AttributeError:
            now = datetime.datetime.now()-datetime.timedelta(days=1)
            self._full_list = pd.read_json(f"http://files.tmdb.org/p/exports/movie_ids_{now.month:02d}_{now.day:02d}_{now.year:04d}.json.gz", compression='gzip', lines=True).sort_values('id')
            return self._full_list

    def save_movie_pictures_pickle(self, start_at:int, max_pics:int = 100, verbose:bool=True ):
        df_list = self.get_full_movie_list()
        df_excerpt = df_list[(df_list['id']>=start_at)]
        result = pd.DataFrame()
        pic_count = 0
        for _,rec in df_excerpt.iterrows():
            # display(rec)
            one_movie = pd.DataFrame.from_records([tmdb._get_dict_from_url(f"https://api.themoviedb.org/3/movie/{rec['id']}", paramsdict={})])
            if not pd.isna(one_movie['poster_path'][0]):
                one_movie['poster'] = tmdb.get_poster(one_movie['poster_path'][0], size='w500')
                result = result.append(one_movie, ignore_index=True)
                pic_count += 1
                if verbose: print('.', end='')
            if pic_count>=max_pics:
                break
        if verbose: print('\nSaving')
        result.to_pickle(f"../data/posters_{start_at:06d}_to_{result.id.max():06d}.pickle.gz", compression='gzip')        
        return result.id.max()+1
        
    def __repr__(self):
        return f'tmdb({str({k[1:]:v for k,v in  tmdb.__dict__.items() if not isinstance(v,(dict, list, pd.core.frame.DataFrame))})})'
