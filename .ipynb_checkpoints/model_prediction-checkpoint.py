from keras.models import load_model
from keras.preprocessing.image import image_utils
from keras.utils import save_img
from segmentation_models.losses import categorical_focal_jaccard_loss, categorical_focal_dice_loss, jaccard_loss
from functions import *
import numpy as np


def predict(  model_name , img_path, mask_path ) :
    
    ###############################################################
    ###########      TAILLE DES IMAGES ET MASQUES       ###########
    ###############################################################
    IMG_SHAPE = {
                "Unet" : ( 128, 256 ),
                "LinkNet" : ( 128, 256 ),
                "PspNet" : ( 144, 288 )
                }
    target_size = IMG_SHAPE[model_name]
    
    ###############################################################
    #########    Chargement du modèle, image et masque    #########
    ###############################################################
    model = load_model(f"models/{model_name}.h5", 
    					custom_objects={"categorical_focal_jaccard_loss": categorical_focal_jaccard_loss,
    									"categorical_focal_dice_loss" : categorical_focal_dice_loss,
    									"focal_loss_plus_jaccard_loss": focal_loss_plus_jaccard_loss,
    									"focal_loss_plus_dice_loss":  focal_loss_plus_dice_loss ,
    									"jaccard_loss" : jaccard_loss ,
    									"ARI_score" : ARI_score, 
    									"mean_IoU" : mean_IoU,
    									"dice_coeff" : dice_coeff,
    									"Tversky_coef" :  Tversky_coef
    									}
    				   )
    img = np.array( image_utils.load_img( img_path , grayscale = False, target_size = target_size, interpolation="lanczos" ),
                    dtype = 'uint8'
                   )  
    mask = np.array( image_utils.load_img( mask_path , grayscale = True, target_size = target_size, interpolation="lanczos" ),
                    dtype = 'uint8'
                    )  

    ################################################################################
    #########    Transformation des segmentation en masque en 8 groupes    #########
    ################################################################################
    img_transformer = Mask_To_8_Groups()
    mask = img_transformer.transform(mask)
    
    
    #################################################################
    #########    Prédiction du masque à partir du modèle    #########
    #################################################################
    mask_pred = model.predict(np.expand_dims(img, axis=0))[0]


    #####################################################
    #########    Evaluation des performances    #########
    #####################################################
    metriques = { "ARI_score" : ARI_score(  mask.ravel() , mask_pred.ravel() ), 
				  "mean_IoU" : mean_IoU(  mask.ravel() , mask_pred.ravel()  ),
				  "dice_coeff" : dice_coeff(  mask.ravel() , mask_pred.ravel()  ),
				  "Tversky_coef" :  Tversky_coef(  mask.ravel() , mask_pred.ravel()  )
				 }
	
	
    ##########################################################################
    #########    Transformation du masque prédit en image couleur    #########
    ##########################################################################
    mask_pred = img_transformer.fit_transform_to_specific_color_set(mask_pred)

    ########################################################################
    #########    Transformation du masque réel en image couleur    #########
    ########################################################################
    mask = img_transformer.fit_transform_to_specific_color_set(mask)
	
    ##############################################
    #########    Sauvegarde du masque    #########
    ##############################################
    save_img( "/".join(img_path.split("/")[:-1])+f"/masque_predit_par_{model_name}.png" , mask_pred)
    
    return metriques, img, mask, mask_pred


