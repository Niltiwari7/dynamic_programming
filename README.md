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
#### Coin problem
