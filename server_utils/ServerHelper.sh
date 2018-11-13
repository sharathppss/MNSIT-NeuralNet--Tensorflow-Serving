#!/bin/bash

dockerFile="Dockerfile"
tfModel="mnsit"
modelServePath="../models/mnist"
modelTrainPath="./training/models"
modelResult="training/result"
config="training/HyperParams.config"

port1=8500
port2=8501
version=1

case $1 in
    build)
        docker build --rm -f $dockerFile -t $tfModel .
        ;;

    train)
        rm -r $modelTrainPath
        ../venv/bin/python3.6 training/Train.py  -config_file $config -model_dir $modelTrainPath -result_dir $modelResult
        ;;

    start)
        docker stop $tfModel

        mkdir -p $modelServePath
        cp -r $modelTrainPath/output/ $modelServePath

        docker run --rm  -v ${PWD}/models_for_serving:/models -e MODEL_NAME=$tfModel -e MODEL_PATH=${PWD}/$modelServePath -p $port1:$port1 -p $port2:$port2 --name $tfModel $tfModel
        ;;

    stop)
        docker stop $tfModel
        ;;
    *)
        echo "Invalid option..."
        ;;
    esac

