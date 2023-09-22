# dynamic_programming

## Linear DP
1. Climbing Stair

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
3. The minimum sum of elements in an array (kadane's algorithm
   
 ```
 #include<bits/stdc++.h>
using namespace std;

int findMinSubarraySum(int *arr, int n) {
    int min_sum = INT_MAX;
    int current_sum = 0;

    for (int i = 0; i < n; i++) {
        current_sum = min(arr[i], current_sum + arr[i]);
        min_sum = min(min_sum, current_sum);
    }

    return min_sum;
}

int main() {
    int n;
    cin >> n;
    int arr[n];
    for (int i = 0; i < n; i++) cin >> arr[i];
    int min_subarray_sum = findMinSubarraySum(arr, n);
    cout << "Smallest sum continuous subarray: " << min_subarray_sum << endl;
}
```
#### Dp on subsequence
```
class Solution {
public:
    int solve(vector<int>& coins, int amount, int i, vector<vector<int>>& dp) {
        if (amount == 0) {
            return 0;
        }
        if (i < 0 || amount < 0) {
            return INT_MAX - 1; // INT_MAX - 1 to avoid overflow
        }
        if (dp[i][amount] != -1) {
            return dp[i][amount];
        }

        // Calculate the minimum number of coins with and without taking the current coin.
        int take = 1 + solve(coins, amount - coins[i], i, dp);
        int not_take = solve(coins, amount, i - 1, dp);

        dp[i][amount] = min(take, not_take);
        return dp[i][amount];
    }

    int coinChange(vector<int>& coins, int amount) {
        int n = coins.size();
        vector<vector<int>> dp(n, vector<int>(amount + 1, -1));
        int result = solve(coins, amount, n - 1, dp);

        // If result is INT_MAX - 1, it means it's not possible to make up the amount.
        return (result == INT_MAX - 1) ? -1 : result;
    }
};

```
#### Target sum
```
class Solution {
public:
    
    // dp vector store precalculated result of <index,sum>
    // max array size 20 and max element sum 1000 and so min element sum 1000
    // makes sum range 0 to  1000+1000 = 2000
    int dp[21][2001];
    
    // recursion from 0 to end of nums array taking +nums[i] and -nums[i] value
    int dfs(int index, vector<int>& nums, int &S, int sum)
    {
        // reached to the end of array 
        // return 1 if S==sum, otherwise return 0
        if(index==nums.size())
            return sum==S?1:0;

        // return precalculted result
        // to avoid negative index we add 1000 with sum
        if(dp[index][sum+1000]!=-1) return dp[index][sum+1000];
        
        int count = 0;
        
        // call recursion for taking both +nums[i] and -nums[i] values
        // and updated running sum as sum+nums[i] and sum-nums[i]
        
        count+= dfs(index+1,nums,S,sum+nums[index]);
        count+= dfs(index+1,nums,S,sum-nums[index]);
        
        return dp[index][sum+1000] = count;
    }
    
    int findTargetSumWays(vector<int>& nums, int S) {
        
        // set -1 to all dp values
        memset(dp,-1,sizeof(dp));
        
        // all possible ways to reach target 
        return dfs(0,nums,S,0);
    }
};
```
#### Coin Change II

```
class Solution {
public:
    int solve(int amount, vector<int>& coins, int i,vector<vector<int>>&dp) {
        if (amount == 0)
            return 1;
        if (amount < 0 || i == coins.size())
            return 0;
        if(dp[i][amount]!=-1)return dp[i][amount];
        // Include the current coin and recurse
        int include = solve(amount - coins[i], coins, i,dp);

        // Exclude the current coin and recurse
        int exclude = solve(amount, coins, i + 1,dp);

        return dp[i][amount]=include + exclude;
    }

    int change(int amount, vector<int>& coins) {
        vector<vector<int>>dp(coins.size(),vector<int>(amount+1,-1));
        return solve(amount, coins, 0,dp);
    }
};
```

#### unbounded knapsack
```
#include <bits/stdc++.h>
int knapsackUtil(vector<int>& wt, vector<int>& val, int ind, int W, vector<vector
<int>>& dp){

    if(ind == 0){
        return ((int)(W/wt[0])) * val[0];
    }
    
    if(dp[ind][W]!=-1)
        return dp[ind][W];
        
    int notTaken = 0 + knapsackUtil(wt,val,ind-1,W,dp);
    
    int taken = INT_MIN;
    if(wt[ind] <= W)
        taken = val[ind] + knapsackUtil(wt,val,ind,W-wt[ind],dp);
        
    return dp[ind][W] = max(notTaken,taken);
}


int unboundedKnapsack(int n, int W, vector<int>& val,vector<int>& wt) {
    
    vector<vector<int>> dp(n,vector<int>(W+1,-1));
    return knapsackUtil(wt, val, n-1, W, dp);
}
```
#### Rod Cutting
```
int cutRodUtil(vector<int>& price, int ind, int N, vector<vector<int>>& dp){

    if(ind == 0){
        return N*price[0];
    }
    
    if(dp[ind][N]!=-1)
        return dp[ind][N];
        
    int notTaken = 0 + cutRodUtil(price,ind-1,N,dp);
    
    int taken = INT_MIN;
    int rodLength = ind+1;
    if(rodLength <= N)
        taken = price[ind] + cutRodUtil(price,ind,N-rodLength,dp);
        
    return dp[ind][N] = max(notTaken,taken);
}


int cutRod(vector<int>& price,int N) {

    vector<vector<int>> dp(N,vector<int>(N+1,-1));
    return cutRodUtil(price,N-1,N,dp);
}
```
#### Longest subsequence-1
```
//{ Driver Code Starts
// Initial Template for C++

#include <bits/stdc++.h>
using namespace std;

// } Driver Code Ends
// User function Template for C++

class Solution{
public:
    int solve(int *arr,int n,int curr,int prev)
    {
        if(curr==n) return 0;
        
        // Length of subsequence including the current element
        int incl = 0;
        if (prev == -1 || arr[curr] - arr[prev] == 1) {
            incl = 1 + solve(arr, n, curr + 1, curr);
        }
        
        // Length of subsequence excluding the current element
        int excl = solve(arr, n, curr + 1, prev);
        
        // Return the maximum of inclusive and exclusive cases
        return max(incl, excl);
    }
    
    int longestSubsequence(int N, int arr[])
    {
        // code here
       vector<int> dp(N,1);
        int ans=INT_MIN;
        for(int i=0;i<N;i++)
        {
            for(int j=0;j<i;j++)
            {
                if(abs(arr[i]-arr[j])==1)
                {
                    dp[i]=max(dp[i],dp[j]+1);
                }
            }
            ans=max(ans,dp[i]);
        }
        return ans;
    }
};
```

### Dp on string
#### print the lcs
```
string findLCS(int n, int m,string &s1, string &s2){
	// Write your code here.
   vector<vector<int>>dp(n+1,vector<int>(m+1));
   for(int j=0;j<=m;j++)dp[0][j]=0;
   for(int i=0;i<=n;i++)dp[i][0]=0;
   for(int i=1;i<=n;i++){
	   for(int j=1;j<=m;j++){
          if(s1[i-1]==s2[j-1])dp[i][j]=1+dp[i-1][j-1];
		  else dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
	   }
   }	
   int len=dp[n][m];
   string ans="";
   for(int i=0;i<len;i++){
	   ans+='$';
   }
   int index=len-1;
   int i=n;
   int j=m;
   while(i>0 && j>0)
   {
	   if(s1[i-1]==s2[j-1]){
		   ans[index]=s1[i-1];
		   index--;
		   i--,j--;
	   }
	   else if(dp[i-1][j]>dp[i][j-1]){
		   i--;
	   }
	   else{
		   j--;
	   }
   }
   return ans;
}
```
#### Longest Common Substring
```
int lcs(string &str1, string &str2){
    // Write your code here.
    int n=str1.size();
    int m=str2.size();
    vector<vector<int>>dp(n+1,vector<int>(m+1,0));

    for(int i=0;i<=n;i++)dp[i][0]=0;
    for(int j=0;j<=m;j++)dp[0][j]=0;
    int ans=0;
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            if(str1[i-1]==str2[j-1]){
                dp[i][j]=1+dp[i-1][j-1];
                ans=max(ans,dp[i][j]);
            }
            else dp[i][j]=0;
        }
    }
    return ans;
}
```
#### Print the longest subsequence
```
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        string w=s;
        reverse(s.begin(),s.end());
        int m=s.size();
        int dp[m+1][m+1];
        for(int i=0;i<m+1;i++){
            dp[i][0]=0;
        }
        for(int i=0;i<m+1;i++){
            dp[0][i]=0;
        }
        for(int i=1;i<m+1;i++){
            for(int j=1;j<m+1;j++){
               if(s[i-1]==w[j-1]){
                  dp[i][j]= 1+dp[i-1][j-1];
               }else{
                   dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
               } 
            }
        }
        return dp[m][m];
    }
};
```
####  Minimum Insertion Steps to Make a String Palindrome
```
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        string w=s;
        reverse(s.begin(),s.end());
        int m=s.size();
        int dp[m+1][m+1];
        for(int i=0;i<m+1;i++){
            dp[i][0]=0;
        }
        for(int i=0;i<m+1;i++){
            dp[0][i]=0;
        }
        for(int i=1;i<m+1;i++){
            for(int j=1;j<m+1;j++){
               if(s[i-1]==w[j-1]){
                  dp[i][j]= 1+dp[i-1][j-1];
               }else{
                   dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
               } 
            }
        }
        return dp[m][m];
    }
    int minInsertions(string s) {
        int n=s.size();
        return n-longestPalindromeSubseq(s);
    }
};
```
####  Delete Operation for Two Strings
```
class Solution {
public:

   int lcs(string s1,string s2,vector<vector<int>>dp)
   {
         int n=s1.size();
         int m=s2.size();

         for(int i=0;i<=n;i++)dp[i][0]=0;
         for(int j=0;j<=m;j++)dp[0][j]=0;

         for(int i=1;i<=n;i++)
         {
             for(int j=1;j<=m;j++){
                 if(s1[i-1]==s2[j-1]){
                     dp[i][j]=1+dp[i-1][j-1];
                 }
                 else{
                     dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
                 }
             }
         }
         return dp[n][m];
         
   }

    int minDistance(string word1, string word2) {
        int n=word1.size();
        int m=word2.size();
        vector<vector<int>>dp(n+1,vector<int>(m+1,-1));
        int ans=lcs(word1,word2,dp);
        return n+m-2*ans;
    }
};
```
#### Distnict subsequence
```
class Solution {
public:
    int solve(string s, string t, int i, int j, vector<vector<int>>& memo) {
        if (j < 0) {
            return 1;  // If t is empty, there is one subsequence (empty string).
        }
        if (i < 0) {
            return 0;  // If s is empty but t is not, there are no subsequence matches.
        }
        if (memo[i][j] != -1) {
            return memo[i][j];  // Return memoized result if available.
        }
        int count = 0;
        if (s[i] == t[j]) {
            // If s[i] matches t[j], we have two choices:
            // 1. Include s[i] in the subsequence.
            // 2. Skip s[i] and continue matching t with s[0...i-1].
            count = solve(s, t, i - 1, j - 1, memo) + solve(s, t, i - 1, j, memo);
        } else {
            // If s[i] doesn't match t[j], we can only skip s[i] and continue matching t with s[0...i-1].
            count = solve(s, t, i - 1, j, memo);
        }
        memo[i][j] = count;  // Memoize the result.
        return count;
    }

    int numDistinct(string s, string t) {
        int m = s.size();
        int n = t.size();
        vector<vector<int>> memo(m, vector<int>(n, -1));
        return solve(s, t, m - 1, n - 1, memo);
    }
};
```
#### Edit distance
```
class Solution {
public:
   int solve(string s1,string s2,int i,int j,vector<vector<int>>&dp)
   {
       if(i<0)return j+1;
       if(j<0)return i+1;
       if(dp[i][j]!=-1)return dp[i][j];
       if(s1[i]==s2[j])
       {
           return 0+solve(s1,s2,i-1,j-1,dp);
       }
      int  a=1+solve(s1,s2,i,j-1,dp);//insert
       int b=1+solve(s1,s2,i-1,j,dp);//delete
       int c=1+solve(s1,s2,i-1,j-1,dp);//replace
       return dp[i][j]=min(a,min(b,c));
   }
    int minDistance(string word1, string word2) {
        int n=word1.size(),m=word2.size();
        vector<vector<int>>dp(n+1,vector<int>(m+1,-1));
      return solve(word1,word2,n-1,m-1,dp);   
    }
};
```
#### wildcard matching
```
class Solution {
public:
    int f(int i,int j, string &s,string &p,vector<vector<int>> &dp)
    {
        if(dp[i][j]!=-1) return dp[i][j];
        if(i==0 && j==0) return dp[i][j]=1;
        if(j==0 && i>0) return dp[i][j]=0;
        if(i==0 && j>0)
        {
            while(j>0)
            {
                if(p[j-1]=='*') j--;
                else return dp[i][j]=0;
            }
            return dp[i][j]=1;
        }
        
        if(s[i-1]==p[j-1] || p[j-1]=='?') return dp[i][j]=f(i-1,j-1,s,p,dp);
        
        if(p[j-1]=='*')
        {
            return dp[i][j] = f(i-1,j,s,p,dp) || f(i,j-1,s,p,dp) ? 1:0;
            //Two cases
            //Consider * as len=0
            //Give one charcter to * and remain at *
            //at next step it will again be decided from both these cases
        }
        return dp[i][j]=0;
        
    }
    
    bool isMatch(string s, string p) {
        int n=s.length(),m=p.length();
        vector<vector<int>> dp(n+1,vector<int>(m+1,-1));
        return f(n,m,s,p,dp);
    }
};
```
### DP ON STOCK
#### Best time to buy sell and stock

```
class Solution {
public:
    int maxProfit(vector<int>& prices) {
    int mini=prices[0];
    int profit=0;
    for(int i=1;i<prices.size();i++){
        int cost=prices[i]-mini;
        profit=max(cost,profit);
        mini=min(mini,prices[i]);
    } 
    return profit;
    }
};
```
#### Best time to buy sell and stock II
```
class Solution {
public:
   int solve(int i,int buy,vector<int>&prices,int n,vector<vector<int>>&dp)
   {
       if(i==n)return 0;
       int profit=0;
       if(dp[i][buy]!=-1)return dp[i][buy];
       if(buy){
           profit=max(-prices[i]+solve(i+1,0,prices,n,dp),0+solve(i+1,1,prices,n,dp));
       }
       else profit=max(prices[i]+solve(i+1,1,prices,n,dp),0+solve(i+1,0,prices,n,dp));
       return dp[i][buy]=profit;
   }
    int maxProfit(vector<int>& prices) {
        int n=prices.size();
        vector<vector<int>>dp(n,vector<int>(2,-1));
      return solve(0,1,prices,n,dp);
    }
};
```
#### Best time to buy sell and stock III
```
class Solution {
public:
     int f(int i,int buy,int cap,vector<int>&prices,vector<vector<vector<int>>>&dp)
     {
         if(i==prices.size()||cap==0)return 0;
         if(dp[i][buy][cap]!=-1)return dp[i][buy][cap];
         int profit=0;
         if(buy){
             profit=max(-prices[i]+f(i+1,0,cap,prices,dp),f(i+1,1,cap,prices,dp));
         }
         else{
             profit=max(prices[i]+f(i+1,1,cap-1,prices,dp),f(i+1,0,cap,prices,dp));
         }
         return dp[i][buy][cap]=profit;
     }
    int maxProfit(vector<int>& prices) {
        int n=prices.size();
        vector<vector<vector<int>>>dp(n,vector<vector<int>>(2,vector<int>(3,-1)));
        return f(0,1,2,prices,dp);
    }
};
```
#### Best time to buy sell and stock IV
```
class Solution {
public:
    int f(int i,int buy,int k,vector<int>&prices,vector<vector<vector<int>>>&dp)
    {
        if(i==prices.size()||k==0)return 0;
        if(dp[i][buy][k]!=-1)return dp[i][buy][k];
        int profit=0;
        if(buy){
            profit=max(-prices[i]+f(i+1,0,k,prices,dp),0+f(i+1,1,k,prices,dp));
        }else{
            profit=max(prices[i]+f(i+1,1,k-1,prices,dp),f(i+1,0,k,prices,dp));
        }
        return dp[i][buy][k]=profit;
    }
  
    int maxProfit(int k, vector<int>& prices) {
        int n=prices.size();
        vector<vector<vector<int>>>dp(n,vector<vector<int>>(2,vector<int>(k+1,-1)));
        return f(0,1,k,prices,dp);
    }
};
```
#### Best time to buy sell and stock with cooldown
```
class Solution {
public:
    int f(int i,int buy,int k,vector<int>&prices,vector<vector<vector<int>>>&dp)
    {
        if(i==prices.size()||k==0)return 0;
        if(dp[i][buy][k]!=-1)return dp[i][buy][k];
        int profit=0;
        if(buy){
            profit=max(-prices[i]+f(i+1,0,k,prices,dp),0+f(i+1,1,k,prices,dp));
        }else{
            profit=max(prices[i]+f(i+1,1,k-1,prices,dp),f(i+1,0,k,prices,dp));
        }
        return dp[i][buy][k]=profit;
    }
  
    int maxProfit(int k, vector<int>& prices) {
        int n=prices.size();
        vector<vector<vector<int>>>dp(n,vector<vector<int>>(2,vector<int>(k+1,-1)));
        return f(0,1,k,prices,dp);
    }
};
```
#### Best time to buy sell and stock with transaction fee
```
class Solution {
public:
    
     int f(int ind,int buy ,vector<int>&prices,int n,vector<vector<int>>&dp,int fee){
        if(ind==n)return 0;
        int profit=0;
        if(dp[ind][buy]!=-1)return dp[ind][buy];
        if(buy){
            profit=max(-prices[ind]-fee+f(ind+1,0,prices,n,dp,fee),0+f(ind+1,1,prices,n,dp,fee));
        }
        else{
            profit=max(prices[ind]+f(ind+1,1,prices,n,dp,fee),0+f(ind+1,0,prices,n,dp,fee));
        }
        return dp[ind][buy]=profit;
    }
    int maxProfit(vector<int>& prices, int fee) {
        int n=prices.size();
       vector<vector<int>>dp(n,vector<int>(2,-1));
       return f(0,1,prices,n,dp,fee);
    }
};
```
### DP on LIS

#### Longest Increasing subsequence
```
class Solution {
public:

    int solve(int i,int prev,vector<int>&nums,vector<vector<int>>&dp)
    {
        if(i==nums.size())return 0;
        if(dp[i][prev+1]!=-1)return dp[i][prev+1];
        int len=solve(i+1,prev,nums,dp);
        if(prev==-1||nums[i]>nums[prev]){
            len=max(len,1+solve(i+1,i,nums,dp));
        }
        return dp[i][prev+1]=len;
    }
    int lengthOfLIS(vector<int>& nums) {
        vector<vector<int>>dp(nums.size(),vector<int>(nums.size()+1,-1));
        return solve(0,-1,nums,dp);
    }
};
```

