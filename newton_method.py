import numpy as np
from math import exp

# サンプルデータの個数
N = 100
# 更新回数の最大値
K = 20
# 終端条件
L = 0.0001
# 乱数のシード
SEED = 1


# 関数f,g
def f(xy, beta0, beta1):
    fz = 0
    for i in range(N):
        p = exp(beta0 + beta1 * xy[i][0])/(1 + exp(beta0 + beta1 * xy[i][0]))
        fz += xy[i][1] - p
    return fz


def g(xy, beta0, beta1):
    gz = 0
    for i in range(N):
        p = exp(beta0 + beta1 * xy[i][0])/(1 + exp(beta0 + beta1 * xy[i][0]))
        gz += xy[i][0] * (xy[i][1] - p)
    return gz


# f,gをそれぞれbeta0とbeta1で偏微分した式
def dfdbeta0(xy, beta0, beta1):
    dfdbeta0z = 0
    for i in range(N):
        p = exp(beta0 + beta1 * xy[i][0]) / (1 + exp(beta0 + beta1 * xy[i][0]))
        dfdbeta0z += -(p * (1 - p))
    return dfdbeta0z


def dfdbeta1(xy, beta0, beta1):
    dfdbeta1z = 0
    for i in range(N):
        p = exp(beta0 + beta1 * xy[i][0])/(1 + exp(beta0 + beta1 * xy[i][0]))
        dfdbeta1z += -(xy[i][0] * p * (1 - p))
    return dfdbeta1z


def dgdbeta0(xy, beta0, beta1):
    dgdbeta0z = 0
    for i in range(N):
        p = exp(beta0 + beta1 * xy[i][0]) / (1 + exp(beta0 + beta1 * xy[i][0]))
        dgdbeta0z += -(xy[i][0] * p * (1 - p))
    return dgdbeta0z


def dgdbeta1(xy, beta0, beta1):
    dgdbeta1z = 0
    for i in range(N):
        p = exp(beta0 + beta1 * xy[i][0]) / (1 + exp(beta0 + beta1 * xy[i][0]))
        dgdbeta1z += -(xy[i][0] * p * (1 - p))
    return dgdbeta1z


def newton_method():
    # 乱数のシードを設定
    np.random.seed(SEED)
    # 一様乱数でNセットの(x, y)の組を生成
    xy = np.random.rand(N, 2)
    # beta0とbeta1の初期値を設定
    beta0, beta1 = 1.0, 1.0
    # 乱数の値でxが0.5を超えていた場合はyを1に、0.5を超えていない場合はyを0にする
    for i in range(N):
        if xy[i][1] >= 0.5:
            xy[i][1] = 1.0
        else:
            xy[i][1] = 0.0

    # 初期化したbeta0とbeta1から初期βベクトル(β0, β1)の転置行列を生成
    beta_vec_k = np.array([beta0, beta1]).T

    # ニュートン法の更新則に従ってβベクトル【beta_vec_k】を更新していく
    for k in range(K):
        print(k, '回目')
        # 元の関数fとgのベクトル(f, g)の転置行列を生成
        fg = np.array([f(xy, beta_vec_k[0], beta_vec_k[1]), g(xy, beta_vec_k[0], beta_vec_k[1])]).T
        # 各関数を各変数で偏微分して得られる行列【m】を生成
        m = np.array([[dfdbeta0(xy, beta_vec_k[0], beta_vec_k[1]), dfdbeta1(xy, beta_vec_k[0], beta_vec_k[1])],
                      [dgdbeta0(xy, beta_vec_k[0], beta_vec_k[1]), dgdbeta1(xy, beta_vec_k[0], beta_vec_k[1])]])
        # 【m】の逆行列【m_inv】を生成
        m_inv = np.linalg.inv(m)
        # ニュートン法の更新則で(k+1)番目のβベクトルを生成
        beta_vec_k_1 = beta_vec_k - np.dot(m_inv, fg)
        print('β(k)', beta_vec_k)
        print('β(k+1)', beta_vec_k_1)
        # k番目のβベクトルの各要素と(k+1)番目のβベクトルの各要素の差をとって差が終端条件L以下であれば探索終了し、超えていれば次の探索を開始
        if abs(beta_vec_k_1[0] - beta_vec_k[0]) <= L and abs(beta_vec_k_1[1] - beta_vec_k[1]) <= L:
            break
        if k == 19:
            print('更新回数20回で終端条件を満たす解が出ませんでした。')
        # βベクトルの(k+1)番目を次のk番目として更新する
        beta_vec_k = beta_vec_k_1


if __name__ == '__main__':
    newton_method()
