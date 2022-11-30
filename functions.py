from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
from enum import Enum
from pydantic import BaseModel
#from tensorflow.python.keras import backend as K
from keras import backend as K
import tensorflow as tf
from matplotlib import colors

###################################################################################
##################     Dice loss et Coefficient de Dice     #######################
###################################################################################

def dice_coeff(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)    
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2.*intersection + smooth)/(K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    
    return 1.0 - dice_coeff(y_true, y_pred)

def binary_dice_entropy(y_true, y_pred):
    
    return categorical_crossentropy(y_true, y_pred) + 3*dice_loss(y_true, y_pred)
    #return CategoricalCrossentropy()(y_true, y_pred) + 3*dice_loss(y_true, y_pred)

##############################################################################
##################     Tversky loss  with beta = 0.5   #######################
##############################################################################

def Tversky_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)    
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2*y_true * y_pred
    denominator = y_true + y_pred 

    return tf.reduce_sum(numerator) / tf.reduce_sum(denominator)

def Tversky_loss(y_true, y_pred):
    return 1.0 - Tversky_coef(y_true, y_pred)

##############################################################################
##################       Intersection sur l'union      #######################
##############################################################################

def mean_IoU(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    score = (intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1. )
    return score

def IOU_good (y_true, y_pred ) :
    y_true = tf.cast(  y_true , tf.float32 )
    y_pred = tf.cast( y_pred, tf.float32)
    
    y_true_f = K.flatten( y_true )
    y_pred_f = K.flatten( y_pred )
    
    i = tf.cast( y_true_f == y_pred_f, tf.float32 )
    j = tf.cast( y_true_f != y_pred_f, tf.float32 )
    
    intersection = tf.reduce_sum( i )
    union = 2*tf.reduce_sum( j ) + intersection

    return intersection/union

def IoU_loss ( y_true, y_pred) :
    
    return 1.0 - mean_IoU(y_true, y_pred)

def Iou_binary_cross_entropy ( y_true, y_pred) :
    
    return categorical_crossentropy( y_true, y_pred ) + 3*IoU_loss( y_true, y_pred )
    #return CategoricalCrossentropy()( y_true, y_pred ) + 3*IoU_loss( y_true, y_pred )



##############################################################################
##################          Segmentation ARI           #######################
##############################################################################

def ARI_score ( y_true, y_pred ) :
    """
    Implementation du score Adjusted Rand score des auteurs Lawrence Hubert  et Phipps Arabic publié dans Journal of Classification en 1985 
    - https://en.wikipedia.org/wiki/Rand_index 
    - Equation 5 de l'article : https://pdfslide.net/documents/lawrence-hubert-and-phipps-arabie-comparing-partitions-1985.html?page=1
    
    Le score permet de mesurer la similitude entre deux partition d'un memê ensemble. ici je l'adpate pour l'évaluation des performances 
    d'une segmentation d'images. la même métrique est disponible sous scikit learn 
    ( https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html) mais s'avère impuissante pour des entrées 
    `y_true` et `y_pred` de type Tensorielle comme celles qui peuvent être fournies par la sortie d'un réseau de convolution.
    
    Liberté vous est donnée de de copier, partager, modifier, améliorer le code à votre gré.
    Paramètres :
    ------------
        y_true, y_pred : Tensor de la forme ( hauteur, largeur , canal ): 
      Auteur :
    -----------
        - Nom     :  Brice KENGNI ZANGUIM
        - E-mail  :  kenzabri2@yahoo.com
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)    
    
    ####################################################################
    ####     Transformation des entrées en un tenseur d'ordre 1    #####
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    #if tf.math.all( y_true_f == y_pred_f ) :
    #    return 1.
    
    nij = tf.math.confusion_matrix(y_true_f,y_pred_f) 
    
    a_i = tf.reduce_sum(nij, axis=0)
    b_j = tf.reduce_sum(nij, axis=1)
    
    a_i_sum = tf.reduce_sum(a_i * (a_i - 1), axis=0)
    b_i_sum = tf.reduce_sum(b_j * (b_j - 1), axis=0)
    n = tf.size(y_pred_f)

    RI = tf.reduce_sum(nij * (nij - 1), axis=(0,1) )
    max_RI = ( a_i_sum + b_i_sum ) / 2
    expected_RI = a_i_sum * b_i_sum / n
    
    RI = tf.cast(RI, tf.float32)
    expected_RI = tf.cast(expected_RI, tf.float32)
    max_RI = tf.cast(max_RI, tf.float32)
    
    return (RI - expected_RI) / (max_RI - expected_RI)

def ARI_loss( y_true ,y_pred ) :
    
    return 1 - ARI_score(y_true, y_pred)

##############################################################################
##################           Entropie mixte            #######################
##############################################################################

def mixt_entropi(y_true =None , y_pred=None, iou_coef=1.4, twersky_coef=1.4,dice_coef=1.2, ari_coef = 0.04) :
    
    IOU = iou_coef*IoU_loss(y_true, y_pred)
    TVERSKY =  twersky_coef*Tversky_loss(y_true, y_pred)
    DICE =  dice_coef*dice_loss( y_true,y_pred )
    ARI = ari_coef*ARI_loss( y_true, y_pred )
    
    return categorical_crossentropy(y_true, y_pred) + IOU + TVERSKY + DICE + ARI
    #return CategoricalCrossentropy()(y_true, y_pred) + IOU + TVERSKY + DICE + ARI


def  focal_loss_plus_jaccard_loss (y_true, y_pred ) :
	return 1 - categorical_focal_jaccard_loss(y_true, y_pred )

def focal_loss_plus_dice_loss (y_true, y_pred ) :
	return 2 - categorical_focal_jaccard_loss(y_true, y_pred )  - dice_loss(y_true, y_pred ) 

########################################################################
##########    Spécification des seuls modèles disponibles     ##########
########################################################################

class AvailableModel( str , Enum ) :
    Unet = "Unet"
    LinkNet = "LinkNet"
    PspNet = "PspNet"


############################################################################################
##########      Classe décrivant le chemin absolu vers l'image et le masque       ##########
############################################################################################

class ModelNameAndImgPath( BaseModel ) :
    model_name : AvailableModel
    img_path : str
    mask_path : str


class Mask_To_8_Groups(BaseEstimator,TransformerMixin):
    
    """
        Description :
        ---------------
        
        Cette classe prends transforme une matrice de taille (n,m) en un tenseur de configuration (n,m,k) où k est le nombre de catégories 
        suivant la troisième dimension k, les composantes sont des vecteurs qui One hot Encode les pixels présents sur la matrice de départ
        
        D'une certaine façon celà équivaut à un One hot Encode des différentes valeurs de pixels présents sur la matrice de départ d'où la
        troisième dimension k qui a pour valeur le nombre de categories
        
        Author :
        -----------
            Name : Brice KENGNI ZANGUIM
            e-mail : kenzabri2@yahoo.com
    """
    ########################################################################################################
    #############        catégorie de groupage par defaut. L'utilisateur peut  fournir         #############
    ############# une autre catergorisation de groupage lors de l'instantiation de la Classe   #############
    categorie_par_defaut =  {
                    'void': [0, 1, 2, 3, 4, 5, 6],
                    'flat': [7, 8, 9, 10],
                    'construction': [11, 12, 13, 14, 15, 16],
                    'object': [17, 18, 19, 20],
                    'nature': [21, 22],
                    'sky': [23],
                    'human': [24, 25],
                    'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]
                }
    
    #############################################################################################################
    #############  Palette de couleurs par defaut pour les classes si l'on veut que chaque classe   #############
    #############  soit représentée par une couleur bien spécifique en RGB. Les couleurs doivent    #############
    #############    suivre le même ordre de definition que dans le dictionnaire de catégories      #############
    palette_couleur_par_defaut =[ 'ivory', 'lightgrey', 'plum', 'olive', 'forestgreen', 'skyblue', 'orangered', 'navy']
    
    def __init__( self, categorie = None, palette_couleur = None):
        ############################################################################################################
        #############  Si une categorisation est fournie lors de l'instantiation alors je l'utilise    #############
        if categorie :
            assert isinstance(categorie, dict)
            self.categorie = categorie
        
        ###################################################################################
        #############  Sinon j'utilise plutot une categorisation par defaut   #############
        else :
            self.categorie = Mask_To_8_Groups.categorie_par_defaut
        self.nom_des_categories = self.categorie.keys()
        self.numero_des_categories = np.array(range(len(self.categorie)) )
        if np.all(palette_couleur) :
            self.palette_couleur = palette_couleur
            assert len(self.palette_couleur) == len(self.categorie)
        else :
            self.palette_couleur = Mask_To_8_Groups.palette_couleur_par_defaut
        

    def fit( self, categorie, palette_couleur = None ):
        self.categorie = categorie
        self.nom_des_categories = categorie.keys()
        self.numero_des_categories = np.array(range(len(self.categorie)) )
        
        if np.all( palette_couleur ) :
            self.palette_couleur = palette_couleur
            assert len(self.palette_couleur) == len(self.categorie)

    
    def transform(self ,img ,y = None ):
        img = np.array( img ) 
        
        #############################################################################
        ############# Initialisation d'un masque de taille (n ,m ,k )   #############
        mask = np.zeros( (img.shape[0], img.shape[1], len( self.nom_des_categories ) ) ) 
        
        ####################################################################################################
        ############# je parcours toute la liste des identifiants `Id` des différents objets   #############
        for Id in range(-1, 34):
            ####################################################################################################
            #############     Je parcours toute la liste des noms de catégories d'objets afin de    ############
            #############   rechercher la catégorie qui associée à l'Id et créer le canal associé   ############
            for pos, cat in enumerate(self.nom_des_categories) :
                if Id in self.categorie[cat]:
                    #mask[:, :, pos] = np.logical_or(mask[:, :, pos], img == Id )
                    mask[:, :, pos][img == Id] = 1
                    break

        return mask
    
    def fit_transform ( self ,img ,categorie = None )  :
        if categorie :
            self.fit( categorie )
        
        return self.transform(img)
    
    def invers_transform( self, mask ) :
        """
        Prends le masque en entrée et reconstruit l'image d'origine au format noir sur blanc
        La petite nuance est qu'il y a à présents autant de niveaux de pixels que de groupes associés à la transformation
        l'image n'est donc pas ramenée au 34 groupes initiaux qui apparaissaient sur l'image originale mais au nombre de 
        groupes réduis associés à la transformation; soit 8 groupes pas defaut.  Pour remontrer aux 34 groupes de départ 
        il faurdrait une astuce plus ingénieuse.
        """
        
        #img = np.zeros((mask.shape[0], mask.shape[1]))
        ###############################################################################################################
        ################  Je me crèe un vectoriser qui donne le pouvoir à une fonction de pouvoir          ############ 
        ################  se transformer en une fonction vectorielle multidimension definie de E vers E    ############ 
        #f = np.vectorize(lambda x: self.numero_des_categories[x] )
        
        ############################################################################################################################
        ####################  Ma fonction vectorielle prends toutes les indexes où se trouvent les maximum              ############ 
        ####################  dans la troisième dimension et y retourne la position le numero de categori  équivalent   ############ 
        if mask.ndim == 3 and mask.shape[-1] == 8 : 
            return np.argmax(mask, axis=2)    # Ceci tire profit du fait que les numéro de catégories sont ordonnés à partir de 0
        
        elif mask.ndim == 3 and mask.shape[-1] == 3  :  # Images au format RGB
            output_mask = np.zeros(mask.shape[0]* mask.shape[1])
            x = mask.reshape((-1,3))
            for classe, couleur in enumerate(self.palette_couleur) :
                output_mask[ np.prod(x == colors.to_rgb( couleur ), axis = 1, dtype=bool) ] = classe
            
            return output_mask.reshape((mask.shape[0],mask.shape[1] ))
        
        #return  f( np.argmax(mask, axis=2)  )
        
    def fit_transform_to_specific_color_set(self,mask, palette_couleur = None ) :
        if palette_couleur : self.palette_couleur = palette_couleur
        
        ##################################################################################################################
        ###############  J'initialise à zéro tous les éléments de l'image RGB qui seras renvoyé en sorti    ##############
        img_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype= float)
                        
        if np.ndim( mask ) == 3 :
            ###################################################################################################################
            ###########    Avant toute chose on se rassure que la profondeur du masque ( sa troisième dimenssion)     #########
            ###########          a la même longueur que le dictionnaires des classes (  nombre de classe )            #########
            # assert mask.shape[-1] == len( self.categorie ) # Uniquement que si la fonction invers_transform ne prends pas les images RGB
            
            ##############################################################################################################
            ###########    Je ramène mon masque de 3 dimensions à un masque de 2 dimension (niveau de gris )     #########
            mask = self.invers_transform(mask)
        
        assert np.ndim(mask) == 2
        for cats in np.unique(mask) :
            for i in [0,1,2] :
                img_rgb[:,:,i][ mask == cats ] = colors.to_rgb( self.palette_couleur[cats] )[i]
        
        return img_rgb
        
        ###Pour chaque cétégorie, donne la valeur de la couleur du pixel (pour R, G et B)
        ###for cat in range(len(self.categorie)):
        ###    for i in [0,1,2] :
        ###        img_rgb[:, :,i] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[i]
            

