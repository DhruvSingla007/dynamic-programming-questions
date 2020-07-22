#include <bits/stdc++.h>

using namespace std;


// knapsack 0-1
int solution(int *w, int *value, int capacity, int size) {
    if (size == 0 || capacity == 0) {
        return 0;
    }

    if (capacity < w[size]) {
        return solution(w, value, capacity, size - 1);
    }

    return max(solution(w, value, capacity, size - 1),
               value[size - 1] + solution(w, value, capacity - w[size - 1], size - 1));
}

int dp_solution(int *w, int *value, int capacity, int size) {
    if (size == 0) {
        return 0;
    }

    int dp[size + 1][capacity + 1];
    for (int i = 0; i <= size; i++) {
        for (int j = 0; j <= capacity; j++) {
            if (i == 0 || j == 0) {
                dp[i][j] = 0;
                continue;
            }

            if (w[i - 1] > j) {
                dp[i][j] = dp[i - 1][j];
            } else {
                dp[i][j] = max(dp[i - 1][j], value[i - 1] + dp[i - 1][j - w[i - 1]]);
            }
        }
    }

    return dp[size][capacity];
}


// l c subsequence
int lcs(string s1, string s2, int l1, int l2) {
    if (l1 == 0 || l2 == 0) {
        return 0;
    }

    if (s1[l1 - 1] == s2[l2 - 1]) {
        return 1 + lcs(s1, s2, l1 - 1, l2 - 1);
    }

    return max(lcs(s1, s2, l1 - 1, l2), lcs(s1, s2, l1, l2 - 1));
}

// matrix chain multiplication
int mcm(int *arr, int i, int j) {
    if (i == j) {
        return 0;
    }

    int res = INT_MAX;
    for (int k = i; k < j; k++) {
        int sub_res = mcm(arr, i, k) + mcm(arr, k + 1, j) + (arr[i - 1] * arr[k] * arr[j]);
        res = min(res, sub_res);
    }

    return res;
}

int dp_mcm(int *arr, int size) {
    if (size <= 2) {
        return 0;
    }

    int dp[size][size];

    for (int i = 1; i < size; i++) {
        dp[i][i] = 0;
    }

    for (int l = 2; l < size; l++) {
        for (int i = 1; i <= size - l; i++) {
            int j = i + l - 1;
            dp[i][j] = INT_MAX;

            for (int k = i; k < j; k++) {
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j] + (arr[i - 1] * arr[k] * arr[j]));
            }
        }
    }

    return dp[1][size - 1];


}

// subset sum problem
int ssp(int *arr, int size, int sum) {
    if (sum == 0) {
        return true;
    }
    if (size == 0) {
        return false;
    }

    if (arr[size - 1] > sum) {
        return ssp(arr, size - 1, sum);
    }

    return ssp(arr, size - 1, sum) || ssp(arr, size - 1, sum - arr[size - 1]);
}

int dp_ssp(int *arr, int size, int sum) {
    if (sum == 0) {
        return true;
    }

    if (size == 0) {
        return false;
    }

    bool dp[size + 1][sum + 1];
    for (int i = 0; i <= size; i++) {
        dp[i][0] = true;
    }

    for (int i = 1; i <= sum; i++) {
        dp[0][i] = false;
    }

    for (int i = 1; i <= size; i++) {
        for (int j = 1; j <= sum; j++) {
            if (arr[i - 1] > j) {
                dp[i][j] = dp[i - 1][j];
            } else {
                dp[i][j] = dp[i - 1][j] || dp[i - 1][j - arr[i - 1]];
            }
        }
    }

    for (int i = 0; i <= size; i++) {
        for (int j = 0; j <= sum; j++) {
            cout << dp[i][j] << " ";
        }
        cout << endl;
    }

    return dp[size][sum];
}

// minimum coin change problem
int mcc(int *coins, int size, int amount) {
    if (amount == 0 || size == 0) {
        return 0;
    }

    int res = INT_MAX;

    for (int i = 0; i < size; i++) {
        if (coins[i] <= amount) {
            int sub_res = mcc(coins, size, amount - coins[i]);
            if (sub_res != INT_MAX) {
                res = min(res, sub_res + 1);
            }
        }
    }

    return res;
}

int dp_mcc(int *coins, int size, int amount) {
    if (size == 0 || amount == 0) {
        return 0;
    }

    vector<int> dp(amount + 1, INT_MAX);
    dp[0] = 0;

    for (int i = 1; i <= amount; i++) {
        for (int j = 0; j < size; j++) {
            if (i >= coins[j]) {
                if (dp[i - coins[j]] != INT_MAX) {
                    dp[i] = min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }
    }

    return dp[amount];

}

// longest increasing subsequence
int lis(int *arr, int size) {
    if (size <= 1) {
        return size;
    }

    vector<int> v;
    v.push_back(1);
    for (int i = 1; i < size; i++) {
        int val = 1;
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i]) {
                val = max(val, v[j] + 1);
            }
        }
        v.push_back(val);
    }

    int maxV = INT_MIN;
    for (int i : v) {
        maxV = max(maxV, i);
    }

    return maxV;
}

int lis_opt(int *arr, int size) {
    if (size <= 1) {
        return size;
    }

    vector<int> v;
    v.push_back(arr[0]);
    for (int i = 1; i < size; i++) {
        if (arr[i] > v[v.size() - 1]) {
            v.push_back(arr[i]);
        } else {
            int index = upper_bound(v.begin(), v.end(), arr[i]) - v.begin();
            v[index] = arr[i];
        }
    }

    return v.size();
}


// cutting rod to maximize profit
int crp(int *val, int n) {
    if (n <= 0) {
        return 0;
    }

    int res = INT_MIN;
    for (int i = 0; i < n; i++) {
        res = max(res, val[i] + crp(val, n - i - 1));
    }

    return res;
}

int dp_crp(int *val, int n) {
    if (n == 0) {
        return 0;
    }

    int dp[n + 1];
    dp[0] = 0;

    for (int i = 1; i <= n; i++) {
        int res = INT_MIN;
        for (int j = 0; j < i; j++) {
            res = max(res, val[j] + dp[i - j - 1]);
        }
        dp[i] = res;
    }

    return dp[n];
}

// Total coin change
int tcc(int *coins, int amount, int size) {
    if (amount == 0) {
        return 1;
    }

    if (size == 0) {
        return 0;
    }

    int res = tcc(coins, amount, size - 1);

    if (coins[size - 1] <= amount) {
        res += tcc(coins, amount - coins[size - 1], size);
    }

    return res;
}

int dp_tcc(int *coins, int amount, int size) {
    if (amount == 0) {
        return 1;
    }
    if (size == 0) {
        return 0;
    }

    int dp[amount + 1][size + 1];

    for (int i = 0; i <= size; i++) {
        dp[0][i] = 1;
    }
    for (int i = 1; i <= amount; i++) {
        dp[i][0] = 0;
    }

    for (int i = 1; i <= amount; i++) {
        for (int j = 1; j <= size; j++) {
            dp[i][j] = dp[i][j - 1];
            if (i >= coins[j - 1]) {
                dp[i][j] += dp[i - coins[j - 1]][j];
            }
        }
    }

    return dp[amount][size];
}

// Egg dropping problem
int edp(int eggs, int floors) {
    if (eggs == 1) {
        return floors;
    }
    if (floors <= 1) {
        return floors;
    }

    int res = INT_MAX;

    for (int i = 1; i <= floors; i++) {
        int sub_res = max(edp(eggs - 1, i - 1), edp(eggs, floors - i));
        res = min(res, sub_res + 1);
    }

    return res;
}

int dp_edp(int eggs, int floors) {
    if (eggs == 1) {
        return floors;
    }
    if (floors <= 1) {
        return floors;
    }

    int dp[eggs + 1][floors + 1];
    for (int i = 0; i <= eggs; i++) {
        dp[i][0] = 0;
        dp[i][1] = 1;
    }

    for (int i = 1; i <= floors; i++) {
        dp[1][i] = i;
    }

    for (int i = 2; i <= eggs; i++) {
        for (int j = 2; j <= floors; j++) {
            dp[i][j] = INT_MAX;
            int sub_res = INT_MIN;
            for (int k = 1; k <= j; k++) {
                sub_res = 1 + max(dp[i - 1][k - 1], dp[i][j - k]);
                dp[i][j] = min(dp[i][j], sub_res);
            }

        }
    }

    return dp[eggs][floors];
}


// maximum sum subarray
pair<int, pair<int, int>> mss(int *arr, int size) {
    int sum = 0;
    int left = 0;
    int right = 0;
    int s = 0;
    int currSum = 0;

    for (int i = 0; i < size; i++) {
        currSum += arr[i];
        if (sum < currSum) {
            sum = currSum;
            left = s;
            right = i;
        }

        if (currSum < 0) {
            s = i + 1;
            currSum = 0;
        }
    }

    return make_pair(sum, make_pair(left, right));
}

// longest common substring
int len_lcs(string s1, string s2, int l1, int l2, int count) {
    if (l1 == 0 || l2 == 0) {
        return count;
    }

    if (s1[l1 - 1] == s2[l2 - 1]) {
        count = len_lcs(s1, s2, l1 - 1, l2 - 1, count + 1);
    }

    count = max(count, max(len_lcs(s1, s2, l1 - 1, l2, 0), len_lcs(s1, s2, l1, l2 - 1, 0)));
    return count;
}

int dp_len_lcs(string s1, string s2) {
    int l1 = s1.length();
    int l2 = s2.length();

    int dp[l1 + 1][l2 + 1];
    int ans = 0;

    for (int i = 0; i <= l1; i++) {
        for (int j = 0; j <= l2; j++) {
            if (i == 0 || j == 0) {
                dp[i][j] = 0;
                continue;
            }
            if (s1[i - 1] == s2[j - 1]) {
                dp[i][j] = 1 + dp[i - 1][j - 1];
                ans = max(ans, dp[i][j]);
            } else {
                dp[i][j] = 0;
            }
        }
    }

    return ans;
}

// word break
bool wb(string s, unordered_set<string> const &set) {
    int size = s.length();
    if (size == 0) {
        return true;
    }

    for (int i = 1; i <= size; i++) {
        if (set.count(s.substr(0, i)) > 0 && wb(s.substr(i, size - i), set)) {
            return true;
        }
    }

    return false;
}

bool dp_wb(string s, unordered_set<string> const &set) {
    int size = s.length();
    if (size == 0) {
        return true;
    }

    vector<bool> dp(size + 1, false);
    dp[0] = true;

    for (int i = 1; i <= size; i++) {
        for (int j = 0; j < i; j++) {
            if (dp[j] && set.count(s.substr(j, i - j)) > 0) {
                dp[i] = true;
                break;
            }
        }
    }

    for (bool b : dp) {
        cout << b << " ";
    }
    cout << endl;

    return dp[size];
}

// stock buy and sell with max k transactions -- Wrong solution :(
int maxProfit(int *prices, int index, int size, int k) {
    if (k == 0 || index >= size) {
        return 0;
    }

    int res = 0;

    for (int i = index; i < size; i++) {
        int sub_res = 0;
        if (prices[i] > prices[index]) {
            sub_res = max(maxProfit(prices, i + 1, size, k),
                          maxProfit(prices, i, size, k - 1) + prices[i] - prices[index]);
        }
        res = max(res, sub_res);
    }

    return res;
}

// -- O(size^2*k)
int dp_maxProfit(int *prices, int size, int k) {
    if (size <= 1 || k == 0) {
        return 0;
    }

    int dp[k + 1][size];
    for (int i = 0; i <= k; i++) {
        for (int j = 0; j < size; j++) {
            if (i == 0 || j == 0) {
                dp[i][j] = 0;
                continue;
            }

            dp[i][j] = dp[i][j - 1];
            int sub_res = INT_MIN;
            for (int x = 0; x < j; x++) {
                sub_res = max(sub_res, -prices[x] + dp[i - 1][x]);
            }
            dp[i][j] = max(dp[i][j], sub_res + prices[j]);
        }
    }

    for (int i = 0; i <= k; i++) {
        for (int j = 0; j < size; j++) {
            cout << dp[i][j] << " ";
        }
        cout << endl;
    }

    return dp[k][size - 1];
}

// -- O(size*k)
int dp_maxProfit_1(int *prices, int size, int k) {
    if (size <= 1 || k == 0) {
        return 0;
    }

    int dp[k + 1][size];
    for (int i = 0; i <= k; i++) {
        int sub_res = INT_MIN;
        for (int j = 0; j < size; j++) {
            if (i == 0 || j == 0) {
                dp[i][j] = 0;
                continue;
            }
            sub_res = max(sub_res, dp[i-1][j-1] - prices[j-1]);
            dp[i][j] = max(dp[i][j - 1], sub_res + prices[j]);
        }
    }

    for (int i = 0; i <= k; i++) {
        for (int j = 0; j < size; j++) {
            cout << dp[i][j] << " ";
        }
        cout << endl;
    }

    return dp[k][size - 1];
}

// maximum sum increasing sub sequence
int msis(int *arr, int size) {
    if (size == 0) {
        return 0;
    }

    vector<int> dp(size, 0);
    dp[0] = arr[0];

    for (int i = 1; i < size; i++) {
        dp[i] = 0;
        int sub_res = 0;
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i]) {
                sub_res = max(sub_res, dp[j]);
            }
        }
        dp[i] = max(arr[i], arr[i] + sub_res);
    }

//    for (int i : dp) {
//        cout << i << " ";
//    }
//    cout << endl;

    int ans = dp[0];
    for (int i : dp) {
        ans = max(ans, i);
    }

    return ans;
}

// Minimum jumps
int min_jumps(int *arr, int size) {
    if (size == 1) {
        return 0;
    }

    int res = INT_MAX;

    for (int i = 0; i <= size - 2; i++) {
        if (i + arr[i] >= size - 1) {
            int sub_res = min_jumps(arr, i + 1);
            if (sub_res != INT_MAX) {
                res = min(res, sub_res + 1);
            }
        }
    }

    return res;
}

int dp_min_jumps(int *arr, int size) {
    if (size <= 1) {
        return 0;
    }

    int dp[size];
    dp[0] = 0;
    for (int i = 1; i < size; i++) {
        dp[i] = INT_MAX;
        for (int j = 0; j < i; j++) {
            if (j + arr[j] >= i) {
                if (dp[j] != INT_MAX) {
                    dp[i] = min(dp[i], dp[j] + 1);
                }
            }
        }
    }

    for (int i : dp) {
        cout << i << " ";
    }
    cout << endl;

    return dp[size - 1];
}

int dp_min_jumps_opt(int* arr, int size){
    if(size <= 1){
        return 0;
    }

    int pos = arr[0];
    int max_reach_pos = arr[0];
    int jumps = 1;

    for(int i = 1; i<size; i++){
        if(pos < i){
            jumps++;
            pos = max_reach_pos;
        }

        max_reach_pos = max(max_reach_pos, arr[i] + i);
    }

    return jumps;
}

// Wildcard matching
bool wcm(string s, string p, int i_s, int i_p) {
    if (i_s == s.length()) {
        for (int i = i_p; i < p.length(); i++) {
            if (p[i] != '*') {
                return false;
            }
        }
        return true;
    }

    if (i_s == s.length() && i_p == p.length()) {
        return true;
    }

    if (p[i_p] == '?' || s[i_s] == p[i_p]) {
        return wcm(s, p, i_s + 1, i_p + 1);
    }

    if (p[i_p] == '*') {
        return wcm(s, p, i_s + 1, i_p) || wcm(s, p, i_s, i_p + 1);
    }

    return false;
}

// optimal strategy in a game
int opg(int *arr, int i, int j, int sum) {
    if (i + 1 == j) {
        return max(arr[i], arr[j]);
    }

    return max(sum - opg(arr, i + 1, j, sum - arr[i]),
               sum - opg(arr, i, j - 1, sum - arr[j]));
}

int opg_2(int *arr, int i, int j) {
    if (j == i + 1) {
        return max(arr[i], arr[j]);
    }


    return max(arr[i] + min(opg_2(arr, i + 1, j - 1),
                            opg_2(arr, i + 2, j)),
               arr[j] + min(opg_2(arr, i + 1, j - 1),
                            opg_2(arr, i, j - 2)));
}

int dp_opg(int *arr, int size) {
    if (size == 0) {
        return 0;
    }

    if (size == 1) {
        return arr[0];
    }

    if (size == 2) {
        return max(arr[0], arr[1]);
    }

    int dp[size][size];

    for (int i = 0; i < size - 1; i++) {
        dp[i][i + 1] = max(arr[i], arr[i + 1]);
    }

    for (int gap = 3; gap < size; gap += 2) {
        for (int i = 0; i + gap < size; i++) {
            int j = i + gap;
            dp[i][j] = max(arr[i] + min(dp[i + 2][j], dp[i + 1][j - 1]),
                           arr[j] + min(dp[i + 1][j - 1], dp[i][j - 2]));
        }
    }

    return dp[0][size - 1];
}


// Maximal Square
int maximalSquare(vector<vector<char>> &matrix) {
    int m = matrix.size();
    int n = matrix[0].size();

    if (m <= 1 || n <= 1) {
        return 0;
    }
    int ans = 0;
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (matrix[i][j] == '0') {
                continue;
            }
            int val = min(matrix[i - 1][j - 1] - '0', min(matrix[i][j - 1] - '0', matrix[i - 1][j] - '0')) + 1;
            matrix[i][j] = (char) (48 + val);
            ans = max(ans, val);
        }
    }

    return ans * ans;
}

// Busting Balloons
int burst_balloons(int *arr, int size) {
    if (size == 0) {
        return 0;
    }
    if (size == 1) {
        return arr[0];
    }

    int dp[size][size];
    memset(dp, 0, sizeof(dp));

    for (int l = 1; l <= size; l++) {
        for (int i = 0; i + l <= size; i++) {
            int j = i + l - 1;
            for (int k = i; k <= j; k++) {
                int left_num = i == 0 ? 1 : arr[i - 1];
                int right_num = j == size - 1 ? 1 : arr[j + 1];

                int left_val = k == i ? 0 : dp[i][k - 1];
                int right_val = k == j ? 0 : dp[k + 1][j];

                dp[i][j] = max(dp[i][j], left_num * arr[k] * right_num + left_val + right_val);
            }
        }
    }

//    for (int i = 0; i < size; i++) {
//        for (int j = 0; j < size; j++) {
//            cout << dp[i][j] << " ";
//        }
//        cout << endl;
//    }

    return dp[0][size - 1];
}

// BST with n keys
int bstNkeys(int n) {
    if (n == 0) {
        return 1;
    }
    if (n <= 2) {
        return n;
    }

    int ans = 0;

    for (int i = 1; i <= n; i++) {
        ans += bstNkeys(i - 1) * bstNkeys(n - i);
    }

    return ans;
}

int dp_bastNkeys(int n) {
    if (n == 0) {
        return 1;
    }
    if (n <= 2) {
        return n;
    }

    int dp[n + 1];
    dp[0] = 1;
    for (int i = 1; i <= n; i++) {
        dp[i] = 0;
        for(int j = 0; j<i; j++){
            dp[i] += dp[j]*dp[i-j-1];
        }
    }

    return dp[n];
}

// maximum square with side as x
int mssx(vector<vector<char>> const &v, int size) {
    pair<int, int> dp[size][size];
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (v[i][j] == 'O') {
                dp[i][j] = make_pair(0, 0);
            } else {
                int horizontal_val = i == 0 ? 1 : dp[i - 1][j].first + 1;
                int vertical_val = j == 0 ? 1 : dp[i][j - 1].second + 1;
                dp[i][j] = make_pair(horizontal_val, vertical_val);
            }
        }
    }

    int ans = 0;

    for (int i = size - 1; i >= 1; i--) {
        for (int j = size - 1; j >= 0; j--) {

            int minV = min(dp[i][j].first, dp[i][j].second);
            while(minV > ans){
                if(dp[i][j - minV + 1].first >= minV && dp[i-minV + 1][j].second >= minV){
                    ans = minV;
                }
                minV--;
            }
        }
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cout << dp[i][j].first << "," << dp[i][j].second << " ";
        }
        cout << endl;
    }

    return ans;
}

int main() {
//    int size;
//    cin >> size;
//
//    int val[size], w[size];
//
//    for (int i = 0; i < size; i++) {
//        cin >> w[i];
//    }
//
//    for (int i = 0; i < size; i++) {
//        cin >> val[i];
//    }
//
//    int capacity;
//    cin >> capacity;
//
//    int ans = solution(w, val, capacity, size);
//    int dp_ans = dp_solution(w, val, capacity, size);
//    cout << ans << endl;
//    cout << dp_ans << endl;

//    string s1, s2;
//    cin >> s1 >> s2;
//
//    int ans = lcs(s1, s2, s1.length(), s2.length());
//    cout << ans << endl;

//    int size;
//    cin >> size;
//    int arr[size];
//    for (int i = 0; i < size; i++) {
//        cin >> arr[i];
//    }
//
//    int ans = mcm(arr, 1, size - 1);
//    cout << ans << endl;
//
//    int dp_ans = dp_mcm(arr, size);
//    cout << dp_ans << endl;

//    int size;
//    cin >> size;
//
//    int arr[size];
//    for (int i = 0; i < size; i++) {
//        cin >> arr[i];
//    }
//
//    int sum;
//    cin >> sum;
//
//    bool ans = ssp(arr, size, sum);
//    cout << ans << endl;
//
//    bool dp_ans = dp_ssp(arr, size, sum);
//    cout << dp_ans << endl;

//    int size;
//    cin >> size;
//
//    int coins[size];
//    for (int i = 0; i < size; i++) {
//        cin >> coins[i];
//    }
//
//    int amount;
//    cin >> amount;
//
//    int ans = mcc(coins, size, amount);
//    cout << ans << endl;
//
//    int dp_ans = dp_mcc(coins, size, amount);
//    cout << dp_ans << endl;

//    int size;
//    cin >> size;
//
//    int arr[size];
//    for (int i = 0; i < size; i++) {
//        cin >> arr[i];
//    }
//
//    int ans = lis(arr, size);
//    cout << ans << endl;
//
//    int dp_ans = lis_opt(arr, size);
//    cout << dp_ans << endl;

//    int size;
//    cin >> size;
//
//    int arr[size];
//    for (int i = 0; i < size; i++) {
//        cin >> arr[i];
//    }
//
//    int ans = crp(arr, size);
//    cout << ans << endl;
//
//    int dp_ans = dp_crp(arr, size);
//    cout << dp_ans << endl;

//    int size;
//    cin >> size;
//
//    int coins[size];
//    for (int i = 0; i < size; i++) {
//        cin >> coins[i];
//    }
//
//    int amount;
//    cin >> amount;
//
//    int ans = tcc(coins, amount, size);
//    cout << ans << endl;
//
//    int dp_ans = dp_tcc(coins, amount, size);
//    cout << dp_ans << endl;

//    int eggs, floors;
//    cin >> eggs >> floors;
//
//    int ans = edp(eggs, floors);
//    cout << ans << endl;
//
//    int dp_ans = dp_edp(eggs, floors);
//    cout << dp_ans << endl;

//    int size;
//    cin >> size;
//
//    int arr[size];
//    for (int i = 0; i < size; i++) {
//        cin >> arr[i];
//    }
//
//    pair<int, pair<int, int>> p = mss(arr, size);
//    cout << p.first << endl;
//
//    for (int i = p.second.first; i <= p.second.second; i++) {
//        cout << arr[i] << " ";
//    }
//    cout << endl;

//    string s1, s2;
//    cin >> s1 >> s2;
//
//    int ans = len_lcs(s1, s2, s1.length(), s2.length(), 0);
//    cout << ans << endl;
//
//    int dp_ans = dp_len_lcs(s1, s2);
//    cout << dp_ans << endl;
//
//    string s;
//    cin >> s;
//    unordered_set<string> set;
//    set.insert("leet");
//    set.insert("code");
//    set.insert("e");
//    set.insert("x");
//    set.insert("man");
//    set.insert("mango");
//    set.insert("icecream");
//    set.insert("and");
//    set.insert("go");
//    set.insert("i");
//    set.insert("like");
//    set.insert("ice");
//    set.insert("cream");

//    bool ans = wb(s, set);
//    cout << ans << endl;
//
//    bool dp_ans = dp_wb(s, set);
//    cout << dp_ans << endl;

//    int size;
//    cin >> size;
//
//    int prices[size];
//    for (int i = 0; i < size; i++) {
//        cin >> prices[i];
//    }
//    int k;
//    cin >> k;

//    int ans = 0;
//    for(int i = 0; i<size; i++){
//        ans = max(ans, maxProfit(prices, i, size, k));
//    }
//    cout << ans << endl;

//    int dp_ans = dp_maxProfit(prices, size, k);
//    cout << dp_ans << endl;
//
//    int dp_ans_1 = dp_maxProfit_1(prices, size, k);
//    cout << dp_ans_1 << endl;

//    int size;
//    cin >> size;
//
//    int arr[size];
//    for (int i = 0; i < size; i++) {
//        cin >> arr[i];
//    }
//
//    int ans = msis(arr, size);
//    cout << ans << endl;

//    int size;
//    cin >> size;
//
//    int arr[size];
//    for (int i = 0; i < size; i++) {
//        cin >> arr[i];
//    }
//
//    int ans = min_jumps(arr, size);
//    cout << ans << endl;
//
//    int dp_ans = dp_min_jumps(arr, size);
//    cout << dp_ans << endl;
//
//    int dp_ans_opt = dp_min_jumps_opt(arr, size);
//    cout << dp_ans_opt << endl;

//    string s, p;
//    cin >> s >> p;
//
//    bool ans = wcm(s, p, 0, 0);
//    cout << ans << endl;

//    int size;
//    cin >> size;
//
//    int arr[size];
//    for (int i = 0; i < size; i++) {
//        cin >> arr[i];
//    }
//
//    int sum = 0;
//    for (int i = 0; i < size; i++) {
//        sum += arr[i];
//    }

//    int ans = opg(arr, 0, size - 1, sum);
//    cout << ans << endl;
//
//    int ans_2 = opg_2(arr, 0, size - 1);
//    cout << ans_2 << endl;
//
//    int dp_ans = dp_opg(arr, size);
//    cout << dp_ans << endl;

//    int m, n;
//    cin >> m >> n;
//
//    vector<vector<char>> matrix;
//    for (int i = 0; i < m; i++) {
//        vector<char> v;
//        for (int j = 0; j < n; j++) {
//            char c;
//            cin >> c;
//            v.push_back(c);
//        }
//        matrix.push_back(v);
//    }
//
//    int ans = maximalSquare(matrix);
//    cout << ans << endl;

//    int size;
//    cin >> size;
//
//    int arr[size];
//    for (int i = 0; i < size; i++) {
//        cin >> arr[i];
//    }
//
//    int ans = burst_balloons(arr, size);
//    cout << ans << endl;

//    int n;
//    cin >> n;
//
//    int ans = bstNkeys(n);
//    cout << ans << endl;

    int size;
    cin >> size;
    vector<vector<char>> v;
    for (int i = 0; i < size; i++) {
        vector<char> sub_v;
        for (int j = 0; j < size; j++) {
            char c;
            cin >> c;
            sub_v.push_back(c);
        }
        v.push_back(sub_v);
    }

    int ans = mssx(v, size);
    cout << ans << endl;


    return 0;
}
