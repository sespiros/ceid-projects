#####################################
# 5070 Seimenis Spyros
#####################################

#!/usr/bin/bash

color[0]='\033[0;31;1;40m' # red 
color[1]='\033[0;32;1;40m' # green 
color[2]='\033[0;33;1;40m' # yellow 
color[3]='\033[0;34;1;40m' # blue 
endcolor='\033[000m'

function func1 {
    read -p 'Client file path[clients.txt]:' file
    
    if [[ ! -f $file || $file = "" ]]; then
        file=clients.txt
    fi

    #tput sc
    echo -e "Loading file $file ...\t\t[${color[2]}WAIT$endcolor]"

    #done=0
    #i=0
    #declare -a loading=('.  ' '.. ' '...' '.. ' '.  ')
    #while [ $done -eq 0 ];do
    #    sleep 1
    #    #tput rc
    #    #echo -ne "\033[1A\033[20D\033[K"
    #    echo -e "\033[1A\033[20D\033[KLoading file $file ${loading[$i]}\t\t[${color[2]}WAIT$endcolor]"
    #    i=$((($i+1)%6))    
    #done
    tmpfile=$file~
    cp $file $tmpfile
    sleep 2
    #tput rc
    echo -ne "\033[1A\033[20D\033[K"
    echo -e "Loading file $file ...\t\t[${color[1]}DONE$endcolor]"
    read -p "Press Enter to return to menu"
}

function func2 {
    read -p "Give a client's code[X0000]:" code
    client=`awk -v code=$code '$0 ~ code {print $0}' $tmpfile`
    if [[ $client = "" ]];then
        echo -e "${color[2]}Client not found.$endcolor"
    else
        echo
        echo $client
        echo
    fi
    read -p "Press Enter to return to menu"
}

function func3 {
    read -p "Give a client's code[X0000]:" ccode
    read -p "Give the new phone number[10-digits]:" num

    awk -F, -v code=$ccode -v number=$num '$0 ~ code{$3=number}{print $1","$2","$3}' $tmpfile > $tmpfile
    read -p "Press Enter to return to menu"
}

function func4 {
    more $tmpfile
    echo
    read -p "Press Enter to return to menu"
}

function func5 {
    tput sc
    echo -e "Saving file $file ...\t\t[${color[2]}WAIT$endcolor]"
    cp $tmpfile $file
    sleep 2
    tput rc
    echo -e "Saving file $file ...\t\t[${color[1]}DONE$endcolor]"
    echo
    read -p "Press Enter to return to menu"
}

function func6 {
    exit
}

function menu {
    clear
    echo -e "${color[3]}============= Phonebook v.1.2 ===============$endcolor"
    echo -e "${color[3]}=============================================$endcolor"
    echo '[1] Load clients file'
    echo '[2] Find telephone of client'
    echo '[3] Change telephone of client'
    echo '[4] View file'
    echo '[5] Save file'
    echo '[6] Exit'
    echo

    read -p 'Give an option[1-6]:' choice

    if [[ $choice < 1 || $choice > 6 ]];then
        echo -e "${color[0]}[!!]Invalid option$endcolor"
        sleep 1
        return
    fi
    func$choice
}

while :; do
    menu
done
