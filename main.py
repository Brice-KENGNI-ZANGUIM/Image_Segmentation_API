#############################################################################
##########  "Parfois nous sommes sur le chemin mais on l'ignore    ##########
##########    jusqu'à ce qu'on atteigne notre destination"         ##########
##########                              Brice KENGNI ZANGUIM.      ##########
#############################################################################


###################################################################
##########    Importation de bibliothèque utilitaires    ##########
###################################################################
import  uvicorn #### ASGI
from fastapi import FastAPI
from model_prediction import predict
from functions import AvailableModel, ModelNameAndImgPath


##############################################################
##########   Création de l'objet Fast API FASPAPI   ##########
##############################################################

FASTAPI = FastAPI()


########################################################################
##########   chemin vers la page d'acceuil de l'application   ##########
########################################################################

@FASTAPI.get("/")
def home() :
    return {"message": "Bonjour et bienvenu sur cette application de segmentation d'images" }


########################################################################
###########     acquisition du nom de modèle à utiliser      ###########
########################################################################
model_name = "Unet"
@FASTAPI.get("/model/{name}")
def get_model_name ( name : AvailableModel ) :
    """
    Recupération du nom du modèle à utiliser pour la prédiction
    ---
    parameters :
     - name : name
       in : query
       type : string
       required = true
    responses :
       200: 
    """
    global model_name
    model_name = name


#########################################################################################
###########     Récupération des chemin absolu vers l'image et le masque      ###########
###########            et éffectuation de la prédiction de masque             ###########
######################################################################################### 

@FASTAPI.post("/predict")
def prediction( datas : ModelNameAndImgPath ) :
    ### Acquisition des données initiales
    data = datas.dict()
    model_name = data["model_name"]
    img_path = data["img_path"]
    mask_path = data["mask_path"]
    
    metriques = predict( model_name , img_path , mask_path )    
    
    return {"model_name" : model_name ,
            "img_path" :img_path ,
            "mask_path" : mask_path ,
            "performance" : metriques , 
            "message" : "Le masque a été prédit avec succès et sauvegardé dans le repertoire de l'image"
           }


########################################################################################
########################################################################################
if __name__ == "__main__" :
    uvicorn.run( FASTAPI, host = "127.0.0.1" , port = 8000 )


# Lancer le programme avec la commande : uvicorn main:FASTAPI --reload
# main indique le nom du fichier à lancer, ici main.py ( fichier python nommé main avec extension .py )
# FASTAPI indique le nom de l'application à utiliser ici FASTAPI, instance de la classe FastAPI
