L'application de segmentation d'images a été déployé sur Streamlit et est accéssible à l'adresse suivante :
https://brice-kengni-zanguim-image-segmentation-a-streamlit-main-budclr.streamlit.app/

A la base l'application est conçue de telle façon qu'un utilisateur puisse :
 - Choisir un modèle pour la segmentation parmi 3 disponibles à savoir : UNET, LINKNET et PSPNET
 - Fournir les chemins d'accès vers l'images et le masque associés : 12 paires (image, masque) sont disponible par defaut et accéssibles avec les chemins "Images_<indice>.png" et "Mask_<indice>.png" où "indice" est un entier pouvant prendre les valeurs entre 0 et 11
 
Cependant le déployement sur Streamlit necessite de créer un repositori sur Github et les sauvegardes .h5 des modèles UNET et LINKNET ont des tailles qui sont suppérieures à 100 Mo; d'un autre côté il n'est pas possible d'éffectuer un push de fichier trop volumineux ( > 100 Mo ) sur github.

En raison de cette contrainte les modèles UNET et LINKNET ne sont donc pas accéssibles et seul le modèle PSPNET peu être utilisé pour éffectuer des segmentations via l'application.