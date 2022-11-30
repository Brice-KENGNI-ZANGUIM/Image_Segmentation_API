#############################################################################
##########  "Parfois nous sommes sur le chemin mais on l'ignore    ##########
##########    jusqu'à ce qu'on atteigne notre destination"         ##########
##########                              Brice KENGNI ZANGUIM.      ##########
#############################################################################


###################################################################
##########    Importation de bibliothèque utilitaires    ##########
###################################################################
from model_prediction import predict
import streamlit as st


############################################################################
##########    Afficher un message pour expliquer l'application    ##########
############################################################################

st.write("## Bienvenu dans l'interface graphique de test de l'application de segmentation réalisée par Brice KENGNI ZANGUIM")
st.write("- Des tests de l'application de segmentation sont possible à partir de fichiers disponibles dans le dossier 'path'")
st.write("- 12 paires (image, masque ) sont disponible par defaut dans le repertoire 'path'")
st.write("- Les noms de ces images et masques sont de la forme : 'Image_[indice].png' et 'Mask_[indice].png' où 'indice' est un entier entre 0 et 11")
st.write("- Vous pouvez par exemple fournir comme chemins :  'path/Image_2.png'  et 'path/Mask_2.png' respectivement pour l'image et le masque associé")


########################################################################
###########     acquisition du nom de modèle à utiliser      ###########
########################################################################
st.write("##### Attention à ne pas utiliser les modèles UNET et LinkNET car les fichiers sauvegardes de ces modèles .h5 n'ont pas pu être chargés sur Github à cause de leurs tailles suppérieures à 100 Mo; ce qui est au delà de la limite admise sur Github. Par conséquent, le seul modèle disponible est le modèle PspNet")
model_name = st.selectbox(
                            label = "1 -  Quel modèle désirez vous utiliser ?",
                            options = ("Unet", "LinkNet", "PspNet"),
                            index = 2 ,
                            
                         ) 


#####################################################################
###########     acquisition chemin d'accès à l'image      ###########
#####################################################################

img_path = st.text_input(
                           label =  "2 - Quel est le chemin d'accès à l'image ? ", 
                           value =  "path/Image_5.png"
                        )

####################################################################
###########     acquisition chemin d'accès au masque     ###########
####################################################################

mask_path = st.text_input(
                           label =  "3 - Quel est le chemin d'accès à le masque? ", 
                           value =  "path/Mask_5.png"
                        )

######################################################################################
###########            Effectuation de la prédiction de masque             ###########
###################################################################################### 

metriques, img, mask, mask_pred = predict( model_name , img_path , mask_path )    

###########################################################################
###########            Affichage de l'image  exacte             ###########
###########################################################################
st.write("- ## Image originale")
st.image( img)

#####################################################################
###########            Affichage des métriques            ###########
#####################################################################
st.write("- ## Métriques de performances \n")
col = st.columns(len(metriques))

i = 0
for metric, val in metriques.items() :
    col[i].metric(
                    label = metric,
                    value = round(float(val ), 4 ),
                    #delta = "0 Unit"
                  )
    i += 1
    
##########################################################################
###########            Affichage du masque  exacte             ###########
##########################################################################
st.write("- ## Masque attendu")
st.image( mask)

##########################################################################
###########            Affichage du masque  prédit             ###########
##########################################################################
st.write("- ## Masque prédit")
st.image( mask_pred )
