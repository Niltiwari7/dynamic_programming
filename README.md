# dynamic_programming

## Linear DP
1. Climing Stair

 ```
   class Solution {
public:
    int f(int n,vector<int>&dp){
        if(n<=0)return 0;
        if(n==1)return 1;
        if(n==2)return 2;
        if(dp[n]!=-1)return dp[n];
        return dp[n]=f(n-1,dp)+f(n-2,dp);
    }
    int climbStairs(int n) {
        vector<int>dp(n+1,-1);
       return f(n,dp);
    }
};
```
2 . Reach a score
```
long long f(long long i,long long k,int arr[]){
   if(i==3)return 0;
   if(k==0)return 1;
   int not_take=f(i+1,k,arr);
   int take=0;
   if(arr[i]<=k)
   take=f(i,k-arr[i],arr);
   return take+not_take;
}
long long int count(long long int n)
{
    int arr[3]={3,5,10};
	return f(0,n,arr);
}
```
