#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#2 Transposition Cipher
// encryption function:
char* encrypt(char* string, size_t length) {
	
	if (length <= 2) {
		char* encE = (char*)calloc(3, sizeof(char));
		encE[0] = string[0];
		encE[1] = string[1];
		return encE;
	}
	
	else if (length >= 8195) {
		printf("Error: Too many characters:");

		void exit();
	}
	else {
		
		
		char temp;
		int beg, fin;
		beg = 0; 
		int middleChar = (length / 2);
		fin = (middleChar - 1);

	while (beg < fin) {
			temp = string[beg];
				string[beg] = string[fin];
				string[fin] = temp;
					beg++; 
					fin--;
		}
		
				beg = middleChar; 
				fin = (length - 1);
	while (beg < fin) {
			temp = string[beg];
				string[beg] = string[fin];
			string[fin] = temp;
					fin--;
					beg++;
		}
		
		char* e1 = encrypt(string, middleChar);
		char* e2 = encrypt(string + middleChar, length - middleChar);		
		char* encE = (char*)calloc(length + 1, sizeof(char));

	int i;

	for (i = 0; i < middleChar; i++)
			encE[i] = e1[i];
	for (i = middleChar; i < length; i++)
			encE[i] = e2[i - middleChar];
	
		
		return encE;
	}
}


//My main function
int main() {
	
	printf("Enter your charaters to be encrypted:");
	
	char* enc = (char*)calloc(8194, sizeof(char));
	char* input = (char*)calloc(8194, sizeof(char));
	
	fgets(input, 8194, stdin);

	int length = strlen(input);
	int f = (length - 1);
	int x;
			enc = encrypt(input, f);
				printf("%d\n", f);
	
			for (x = 0; x < f; x++)
				printf("%c", enc[x]);

}
