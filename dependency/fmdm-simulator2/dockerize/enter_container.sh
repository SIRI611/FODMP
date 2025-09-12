#!/bin/bash

headless=1
gui=2

echo "Please enter the character corresponding to the container you would like to enter."
echo "("$headless") headless_pineapple"
echo "("$gui") gui_pineapple"
read CONTAINER

while [[ $CONTAINER -ne $headless ]] && [[ $CONTAINER -ne $gui ]]
do
  echo "Please enter a valid number."
  read CONTAINER
done

docker ps -a

if [[ $CONTAINER -eq $headless ]]
then
  echo "Is headless_pineapple up? [Y/n]"
  read RUNNING
  if [[ "$RUNNING" == "Y" ]] || [[ "$RUNNING" == "y" ]]
  then
    echo "Launching..."
    docker exec -it headless_pineapple bash
  else
    echo "Please run the headless_pineapple container first."
  fi
elif [ $CONTAINER -eq $gui ]
then
  echo "Is gui_pineapple up? [Y/n]"
  read RUNNING
  if [[ "$RUNNING" == "Y" ]] || [[ "$RUNNING" == "y" ]]
  then
    echo "Launching..."
    docker exec -it gui_pineapple bash
  else
    echo "Please run the gui_pineapple container first."
  fi
fi
