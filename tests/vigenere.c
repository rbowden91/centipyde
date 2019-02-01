#include <stdio.h>
#include <cs50.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

int main(int argc, string argv[]){
    if(argv[1]==NULL||argc==0){//returns error if there is no argument
        printf("Idiot!");
        return 1;
    }
    for(int j=0,leng=strlen(argv[1]);j<leng;j++){//returns error if keyword has stuff other than letters
        if((isalpha(argv[1][j]))==false){
            printf("Idiot!");
            return 1;
        }
    }
    string key=argv[1];
    string words=GetString();
    int keyCount=0;
    for(int i =0,h=strlen(words);i<h;i++){

        if(isupper(words[i])){//Uppercase
               if(islower(key[keyCount])){//Bumps up by key position in alphabet. Does toupper just for comfort
                   key[keyCount]=toupper(key[keyCount]);
               }
               int keyBump=((int)key[keyCount])-65;//sets value to bump letter by.

               int letter=(int)words[i];

               if(letter+keyBump>90){//wraparound case
                  letter=64+((letter+keyBump)%90);
               }
               
               else{
               letter=letter+keyBump;
               }

               words[i]=letter;
               if(keyCount==(strlen(key)-1)){
                   keyCount=0;}
               else{
                   keyCount++;}
               }
        if(islower(words[i])){
               if(islower(key[keyCount])){//Bumps up by key position in alphabet. Does toupper just for comfort
                   key[keyCount]=toupper(key[keyCount]);
               }
               int keyBump=((int)key[keyCount])-65;

               int letter=toupper((int)words[i]);//Does this also for comfort.

               if(letter+keyBump>90){//wraparound case
                  letter=64+((letter+keyBump)%90);
               }
               else{
               letter=letter+keyBump;
               }

               words[i]=tolower(letter);//corrects the thing done for comfort
               if(keyCount==(strlen(key)-1)){
                   keyCount=0;}
               else{
                   keyCount++;}
            }

        
    }
 
        printf("%s\n",words);


}
