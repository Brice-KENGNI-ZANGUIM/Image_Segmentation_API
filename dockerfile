########################################################################################
################                   Image de base Ubuntu                 ################
########################################################################################
FROM ubuntu:22.04.1

########################################################################################
################                     Créateur du docker                 ################
########################################################################################
MAINTAINER Brice KENGNI ZANGUIM <kenzabri2@yahoo.com>

########################################################################################
######## Copier le contenu du repertoire actuel dans le repertoir  de reférence ########
########################################################################################
COPY . /usr/app/

########################################################################################
########        repertoire de base ou repertoire de reférence de travail        ########
########################################################################################
WORKDIR /usr/app/

########################################################################################
################        Port sur lequel sera exposé l'application       ################
########################################################################################
EXPOSE 8000

########################################################################################
####### commandes à exécuter, en général pour l'installation de l'environnement ########
########################################################################################
RUN pip install requirements.txt

########################################################################################
########    Indiquer par quelle commande le Docker doit lancer l'application    ########
########################################################################################
CMD uvicorn main:FASTAPI --reload
