import java.util.Scanner;
import java.math.BigInteger;

class Main {                          

// use BigInteger for every value
//to compile do "javac [filename]"
// to run do "java [classname]"

  public static void main(String[] args) {

  BigInteger b1,b2,b3;
  b1 = new BigInteger("2"); 
  b2 = new BigInteger("11"); 
  int p1 = 1010;
  int p2 = 2020;

  b3=b1.pow(p1);

  BigInteger faa = b3.add(b2.pow(p2));
  System.out.println(faa);

  // list all 3 digit primes
  int [] A = new int[] {101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 
  163, 167, 173, 179, 181, 191, 193, 197, 199,211,223,227,229,
  233,239,241,251,257,263,269,271,277,281, 
  283,293,307,311,313,317,331,337,347,349, 
  353,359,367,373,379,383,389,397,401,409, 
  419,421,431,433,439,443,449,457,461,463, 
  467,479,487,491,499,503,509,521,523,541, 
  547,557,563,569,571,577,587,593,599,601, 
  607,613,617,619,631,641,643,647,653,659, 
  661,673,677,683,691,701,709,719,727,733, 
  739,743,751,757,761,769,773,787,797,809, 
  811,821,823,827,829,839,853,857,859,863, 
  877,881,883,887,907,911,919,929,937,941, 
  947,953,967,971,977,983,991,997
  };

  for (int i = 0; i < A.length; i++){
    if ( faa.mod(BigInteger.valueOf(A[i])) == BigInteger.valueOf(0)) System.out.println(A[i]); 

    }
  } 
}
