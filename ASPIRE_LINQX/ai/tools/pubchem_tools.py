import requests
from ASPIRE_LINQX.ai.tools.constants import *
import pprint
from langchain.tools import tool

# TODO: create custom error
# TODO: add functionality in these functions to make use of inCHi and name

# ASK: are IDs always of type int?
# Name and smiles is not allowed for multiple inp, do you want a check for that? or is the generic error okay?
def convert_to_string(l_ids):
    if isinstance(l_ids, list):
        l_ids = [str(l) for l in l_ids] #Do we need this if the LLM is calling it?
        l_ids = ','.join(l_ids)
    else: 
        l_ids = str(l_ids)
    return l_ids

def get_request(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else: 
        print('Error: \n Staus code: {}\nMessage: {}\n URL created: {}'.format(response.status_code, response.text, url))
        return None  # or return some error indicating that the request URL is not valid

@tool
def get_sids_from_cid(inp:list, inp_type:str='compound') -> dict:
    '''get SIDs (Substance IDs) for a given CID (Compound ID)'''
    try:
        inp = convert_to_string(inp)
        
        url = '/'.join([PUBCHEM, inp_type, 'cid', inp, 'sids', OUTPUT_FORMAT])
        
        return get_request(url)
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None  

@tool
def get_cids_from_sid(inp:list, inp_type:str='substance') -> dict:
    '''get CID (Compound IDs) for a given SID (Substance ID)'''
    try:
        inp = convert_to_string(inp)
        url = '/'.join([PUBCHEM, inp_type, 'sid', inp, 'cids', OUTPUT_FORMAT])
    
        return get_request(url)
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None 
    
@tool
def get_synonym(inp, inp_format, inp_type) -> dict:
    '''
    Get Synonym of a substance or compound.
    inp: list of identifiers
    inp_format: can be name, sid, cid, smiles
    inp_type: 'compound' if inp_format is cid, name or smiles of compound or 'substance' if inp_format is sid, name, smiles of substance'''
    try: 
        inp = convert_to_string(inp)
        url = '/'.join([PUBCHEM, inp_type, inp_format, inp, 'synonyms', OUTPUT_FORMAT])
        return get_request(url)
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None 
    
@tool
def get_description(inp, inp_format, inp_type) -> dict:
    '''Get description of a substance or a compound, for assay description, use get_assay_description() instead

    inp: list of identifiers
    inp_format: name, sid, cid, smiles 
    inp_type: 'compound' if inp_format is cid, name or smiles of compound or 'substance' if inp_format is sid, name, smiles of substance'''
    try: 
        inp = convert_to_string(inp)
        url = '/'.join([PUBCHEM, inp_type, inp_format, inp, 'description', OUTPUT_FORMAT])
        return get_request(url)
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None 
    

def get_classification_nodes(output_format, hnid):
    '''The output identifier type is case-insensitive and must be one of: cid, compound; sid, 
    substance; aid, bioassay; patent; pmid, pubmedid; doi; gene, geneid; protein; taxonomy, 
    taxonomyid; pathway, pathwayid; disease, diseaseid; or cell, cellid.'''
    try:
        url =  '/'.join([PUBCHEM, 'classification/hnid', str(hnid), output_format, OUTPUT_FORMAT])
        return get_request(url)
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None 

@tool   
def get_compound_property_table(inp, inp_format, inp_type, property_list):
    # works for name, cid, cids, smiles
    '''Get a table of properties for a given compound or substance.
    
    inp: list of identifiers
    inp_format: name, sid, cid, smiles
    inp_type: 'compound' if inp_format is cid, name or smiles of compound or 'substance' if inp_format is sid, name, smiles of substance
    property_list: 
    '''
    try:
        if len(property_list) == 0 or not set(property_list).issubset(set(PROPERTIES)):
            raise ValueError("Invalid property list")
        inp = convert_to_string(inp)
        property_list = convert_to_string(property_list)
        print(property_list)
        url =  '/'.join([PUBCHEM,  inp_type, inp_format, inp, 'property', property_list, OUTPUT_FORMAT])  
        return get_request(url)
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None    

@tool  
def get_assay_description(aid: int) -> dict:
    "Get Assay description, protocol and comment on the scores for a given assay id."
    try:
        inp = convert_to_string(aid)
        url = '/'.join([PUBCHEM, "assay", "aid", inp, 'description', OUTPUT_FORMAT])
        print(url)
        res = get_request(url)
        # pp.pprint(res)
        specific_pairs = {}
        res = res['PC_AssayContainer'][0]['assay']['descr']

        # Iterate over the dictionary
        for key in ["description", "protocol", "comment"]:
        # Check if the key exists in the dictionary
            if key in res:
                # Add the key-value pair to the specific_pairs dictionary
                specific_pairs[key] = res[key]

        return specific_pairs
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None   

# TODO: add default number of assay IDs, try with docstring instruction as well.
@tool
def get_assay_id_from_smiles(smiles: str) -> str:
    '''Gives you the assay ID (aid) for a single smiles string of a compound. 
    If the user specifies that the item is a substance, then ask the user to enter SMILES for a compound
    
    Note: An assay is a process of analyzing a compound to determine its composition or quality.
    This function gives you all the assays that have used the given compound for testing.'''
    try:
        url = '/'.join([PUBCHEM, 'compound', 'smiles', smiles, 'aids', OUTPUT_FORMAT]) 
        print(url)
        res = get_request(url)
        return res['InformationList']['Information'][0]['AID']
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None   
    
@tool
def get_assay_name_from_aid(aid: list) -> str:
    "Gives a dictionary of names for each assay ID (aid)"
    try:
        inp = convert_to_string(aid)
        url = '/'.join([PUBCHEM, 'assay', 'aid', inp, 'description', OUTPUT_FORMAT]) 
        res = get_request(url)
        # pp.pprint(res)
        res = res['PC_AssayContainer']
        names = dict()
        for i, desc in enumerate (res):
            id = str(desc['assay']['descr']['aid']['id'])
            name = desc['assay']['descr']['name']
            names[id] = name
            print('names', names[id])
        return names
    except Exception as e:
        print('An error occurred:', e)
        # return e
        return None 


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)
    result = get_assay_name_from_aid([9123, 3332, 5358,6683])
    # result = get_description(inp=1, inp_format='sid', inp_type='substance')
    print(type(result))
    print(result)
    if result:
        pp.pprint("Result: {}".format(result))
    else:
        pp.pprint("Error occurred while fetching data.")
        raise error

    # Getting SIDs from a single CID:
    try: 
        result = get_sids_from_cid(2244)
        if result:
            pp.pprint("Result: {}".format(result))
        else:
            pp.pprint("Error occurred while fetching data.")
            # raise error

        result = get_sids_from_cid([2244, 2245, 2246])
        if result:
            pp.pprint("Result: {}".format(result))
        else:
            pp.pprint("Error occurred while fetching data.")

    except:
        print("errror")