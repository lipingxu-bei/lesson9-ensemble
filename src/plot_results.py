# -*- coding: utf-8 -*-
"""Regenerate Fig1–Fig5 from coursework CSV and the student training/validation data."""
import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
from matplotlib.ticker import PercentFormatter, MaxNLocator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score

P=Path(__file__).resolve().parents[1]
OUT=P/'figures'
BASE=P/'results'
SEED=1010
CN='Noto Sans CJK JP'
plt.rcParams.update({'font.family':CN, 'axes.unicode_minus':False,'font.size':9.2,'axes.titlesize':10.5,
    'axes.labelsize':9,'xtick.labelsize':8.2,'ytick.labelsize':8.2,'legend.fontsize':8,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'xtick.major.size':3,'ytick.major.size':3,
    'figure.facecolor':'white','savefig.facecolor':'white','grid.linewidth':.5})
BLUE='#226a88'; ORANGE='#d98456'; GREEN='#438e81'; NAVY='#273b50'; GRAY='#8a96a0'; LT='#eff4f6'; PURPLE='#7663a1'; RED='#b96262'
COLORS={'Single tree':GRAY,'Bagging':ORANGE,'Random forest':BLUE,'AdaBoost':PURPLE,
        'Gradient boosting':GREEN,'Logistic regression':NAVY,'RBF-SVM':RED}
CHN={'Single tree':'单棵树','Bagging':'Bagging','Random forest':'随机森林','AdaBoost':'AdaBoost',
        'Gradient boosting':'梯度提升','Logistic regression':'逻辑回归','RBF-SVM':'RBF-SVM'}

def rd(name):return pd.read_csv(BASE/name)
def tag(ax,label):ax.text(-.10,1.08,label,transform=ax.transAxes,va='top',ha='left',fontweight='bold',fontsize=12,color=NAVY)
def sty(ax,axis='y'):
    ax.grid(axis=axis,color='#e1e8ec',alpha=.9);ax.set_axisbelow(True)
def save(fig,name):
    OUT.mkdir(exist_ok=True,parents=True)
    fig.savefig(OUT/(name+'.png'),dpi=260,bbox_inches='tight',pad_inches=.09)
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.09)
    plt.close(fig)


def main() -> None:
    global OUT, BASE, P
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=P, help='Root containing data/student, results, figures')
    parser.add_argument('--figures-dir', type=Path, default=None)
    parser.add_argument('--results-dir', type=Path, default=None)
    parser.add_argument('--auc-bootstrap-repeats', type=int, default=0,
                        help='Optional expensive diagnostic; 0 skips CI computation')
    args=parser.parse_args()
    P=args.project_root.resolve()
    OUT=(args.figures_dir or P/'figures').resolve()
    BASE=(args.results_dir or P/'results').resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    # --- Figure 1 ---
    fig=plt.figure(figsize=(7.0,4.65)); gs=fig.add_gridspec(2,2,wspace=.39,hspace=.62)
    ax=fig.add_subplot(gs[0,0]); tag(ax,'a')
    names=['训练集','验证集','锁定测试集']; rows=[341,114,114]; pos=[127,43,None]
    for i,(n,k) in enumerate(zip(names,rows)):
        ax.barh(i,k,color=[BLUE,ORANGE,'#c3cdd3'][i],height=.55)
        ax.text(k+7,i,f'{k} 条',va='center',fontsize=8,color=NAVY)
    ax.set_yticks(range(3),names);ax.invert_yaxis();ax.set_xlim(0,405);ax.set_xticks([0,100,200,300,400]);ax.set_xlabel('样本数')
    ax.set_title('研究数据与评估划分',loc='left',pad=9);sty(ax,'x')
    ax=fig.add_subplot(gs[0,1]);tag(ax,'b')
    ax.text(.5,.84,'30 个形态特征',ha='center',va='center',color=NAVY,fontsize=11,fontweight='bold',transform=ax.transAxes)
    for x0,label,fc in [(.03,'训练：学习参数',BLUE),(.365,'验证：比较模型',ORANGE),(.70,'测试：暂不触碰','#aebbc2')]:
        r=patches.FancyBboxPatch((x0,.35),.29,.29,boxstyle='round,pad=0.018,rounding_size=.045',transform=ax.transAxes,facecolor=fc,edgecolor='none')
        ax.add_patch(r);ax.text(x0+.145,.495,label,ha='center',va='center',fontsize=7.7,color='white' if x0<.7 else NAVY,transform=ax.transAxes)
    for x in [.325,.66]: ax.annotate('',xy=(x+.035,.5),xytext=(x-.01,.5),xycoords=ax.transAxes,textcoords=ax.transAxes,arrowprops=dict(arrowstyle='->',color=GRAY,lw=1.2))
    ax.text(.5,.13,'同一份验证数据用于画图与模型比较\n不用于最终独立泛化结论',ha='center',va='center',color='#596977',fontsize=7.6,transform=ax.transAxes)
    ax.set_axis_off(); ax.set_title('避免信息泄露的基本协议',loc='left',pad=9)
    ax=fig.add_subplot(gs[1,0]);tag(ax,'c')
    boot=rd('C_bootstrap_1000.csv');v=boot.unique_fraction.values
    ax.hist(v,bins=np.arange(.565,.702,.01),color=BLUE,alpha=.78,edgecolor='white',linewidth=.5)
    ax.axvline(v.mean(),color=ORANGE,lw=1.8,ls='--',label=f'均值 {v.mean():.3f}')
    ax.axvline(1-np.exp(-1),color=NAVY,lw=1,ls=':',label='理论约 0.632')
    ax.set_xlabel('每轮不同训练样本的比例');ax.set_ylabel('模拟次数');ax.legend(frameon=False,loc='upper left',fontsize=7.4)
    ax.set_title('1,000 次 Bootstrap 的样本覆盖',loc='left',pad=9);sty(ax)
    ax=fig.add_subplot(gs[1,1]);tag(ax,'d')
    root=rd('A_bootstrap_roots.csv').root_feature.value_counts()
    root.index=[s.replace('worst ','W. ').replace('mean ','M. ') for s in root.index]
    ax.barh(root.index[::-1],root.values[::-1],color=GREEN,height=.55)
    for i,k in enumerate(root.values[::-1]):ax.text(k+.1,i,str(k),va='center',color=NAVY,fontsize=8)
    ax.set_xlim(0,max(root)+1.6);ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('出现次数 / 12 次');ax.set_title('浅树根节点会随抽样改变',loc='left',pad=9);sty(ax,'x')
    fig.subplots_adjust(top=.94);save(fig,'Fig1_研究设计与Bootstrap')

    # --- Figure 2 ---
    fig=plt.figure(figsize=(7,5.0));gs=fig.add_gridspec(2,2,wspace=.38,hspace=.62)
    ax=fig.add_subplot(gs[0,0]);tag(ax,'a');dd=rd('A_tree_depth.csv');x=np.arange(5)
    ax.plot(x,dd.train_acc,color=BLUE,marker='o',label='训练准确率',lw=1.9)
    ax.plot(x,dd.validation_acc,color=ORANGE,marker='s',label='验证准确率',lw=1.9)
    ax.set_xticks(x,['1','2','3','4','不限']);ax.set_ylim(.89,1.012);ax.set_ylabel('Accuracy');ax.set_xlabel('决策树最大深度')
    ax.legend(frameon=False,loc='lower right');ax.set_title('树越深，训练越好；验证未必越好',loc='left',pad=9);sty(ax)
    ax=fig.add_subplot(gs[0,1]);tag(ax,'b');cur=rd('D_tree_count.csv')
    for model in ['Bagging','Random forest']:
     q=cur[cur.model==model];ax.plot(q.n_trees,q.auc,marker='o',markersize=4,lw=1.9,color=COLORS[model],label=CHN[model])
    ax.set_xscale('log');ax.set_xticks([1,10,50,100,300],['1','10','50','100','300']);ax.set_ylim(.91,1.01)
    ax.set_xlabel('集成中树的数量');ax.set_ylabel('验证 ROC-AUC');ax.legend(frameon=False,loc='lower right')
    ax.set_title('10–50 棵树后改善进入平台',loc='left',pad=9);sty(ax)
    ax=fig.add_subplot(gs[1,0]);tag(ax,'c');val=rd('D_F_validation_models.csv').set_index('model')
    selected=['Single tree','Bagging','Random forest'];xs=np.arange(3)
    ax.bar(xs-.18,val.loc[selected,'roc_auc'],width=.35,color=ORANGE,label='ROC-AUC')
    ax.bar(xs+.18,val.loc[selected,'balanced_accuracy'],width=.35,color=BLUE,label='平衡准确率')
    ax.set_ylim(.88,1.013);ax.set_xticks(xs,[CHN[s] for s in selected]);ax.legend(frameon=False,ncol=2,loc='upper center',fontsize=7.5)
    ax.set_title('集成比单棵树的 AUC 更高',loc='left',pad=9);sty(ax)
    ax=fig.add_subplot(gs[1,1]);tag(ax,'d');sta=rd('D_G_prediction_stability.csv').set_index('model').loc[selected]
    xs=np.arange(3);bars=ax.bar(xs,sta.flip_rate*100,color=[GRAY,ORANGE,BLUE],width=.57)
    for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+.8,f'{b.get_height():.1f}%',ha='center',fontsize=8,color=NAVY)
    ax.set_ylim(0,30);ax.set_ylabel('至少发生过一次类别翻转的样本（%）')
    ax.set_xticks(xs,[CHN[s] for s in selected]);ax.set_title('随机森林的预测更不容易“变卦”',loc='left',pad=9);sty(ax)
    save(fig,'Fig2_单树与集成效果')

    # compute ROC curves + bootstrap uncertainty for fixed fitted models
    DAT=P/'data'/'student'; train=pd.read_csv(DAT/'train.csv');valid=pd.read_csv(DAT/'validation.csv')
    features=[c for c in train.columns if c not in ('sample_id','malignant')]
    X=train[features];Y=train.malignant;XV=valid[features];YV=valid.malignant.values
    mods={
        'Single tree':DecisionTreeClassifier(max_depth=4,random_state=SEED),
        'Bagging':BaggingClassifier(estimator=DecisionTreeClassifier(random_state=SEED),n_estimators=100,random_state=SEED,n_jobs=1),
        'Random forest':RandomForestClassifier(n_estimators=100,max_features='sqrt',random_state=SEED,n_jobs=1),
        'AdaBoost':AdaBoostClassifier(n_estimators=50,random_state=SEED),
        'Gradient boosting':GradientBoostingClassifier(n_estimators=100,random_state=SEED),
        'Logistic regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,random_state=SEED)),
        'RBF-SVM':make_pipeline(StandardScaler(),SVC(C=1,kernel='rbf',random_state=SEED))}
    raw={};ypred={};curves={}
    for name,m in mods.items():
     m.fit(X,Y);s=m.decision_function(XV) if hasattr(m,'decision_function') else m.predict_proba(XV)[:,1]
     raw[name]=s;ypred[name]=m.predict(XV)
     fpr,tpr,_=roc_curve(YV,s);curves[name]=(fpr,tpr)

    # Diagnostic only, optional: not part of the five figures or final test evaluation.
    ci_rows=[]
    if args.auc_bootstrap_repeats:
        rng=np.random.default_rng(SEED+67)
        ids=[rng.choice(len(YV),len(YV),replace=True)
             for _ in range(args.auc_bootstrap_repeats)]
        for name,score in raw.items():
            vals=[roc_auc_score(YV[ii], score[ii]) for ii in ids
                  if len(np.unique(YV[ii])) == 2]
            ci_rows.append({'model':name,'auc_valid':roc_auc_score(YV,score),
                            'bootstrap_val_low':np.quantile(vals,.025),
                            'bootstrap_val_high':np.quantile(vals,.975)})
        pd.DataFrame(ci_rows).to_csv(OUT/'validation_auc_bootstrap_diagnostic.csv',index=False)

    # --- Figure 3 ---
    fig=plt.figure(figsize=(7.0,5.1));gs=fig.add_gridspec(2,2,wspace=.42,hspace=.57)
    ax=fig.add_subplot(gs[0,0]);tag(ax,'a')
    for name in ['Single tree','Bagging','Random forest','Logistic regression','RBF-SVM']:
     fpr,tpr=curves[name];ax.plot(fpr,tpr,color=COLORS[name],lw=1.5,label=f'{CHN[name]} ({roc_auc_score(YV,raw[name]):.3f})')
    ax.plot([0,1],[0,1],ls=':',color='#bec7cd',lw=.7)
    ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.set_xlabel('假阳性率');ax.set_ylabel('真阳性率')
    ax.set_title('验证集 ROC 曲线',loc='left',pad=9);ax.legend(frameon=False,loc='lower right',fontsize=7)
    ax=fig.add_subplot(gs[0,1]);tag(ax,'b')
    cv=rd('F_development_cv.csv').set_index('model'); vv=rd('D_F_validation_models.csv').set_index('model')
    order=['Single tree','Bagging','Random forest','Gradient boosting','Logistic regression','RBF-SVM']
    ind=np.arange(len(order));ax.errorbar(cv.loc[order,'cv_auc_mean'],ind+.12,xerr=cv.loc[order,'cv_auc_sd'],fmt='o',color=BLUE,capsize=2,ms=4,label='5 折 CV 均值 ± SD')
    ax.scatter(vv.loc[order,'roc_auc'],ind-.13,s=23,marker='D',color=ORANGE,label='固定验证集')
    ax.set_yticks(ind,[CHN[n] for n in order]);ax.invert_yaxis();ax.set_xlim(.87,1.016);ax.set_xlabel('ROC-AUC');ax.set_title('不能只按一次验证集排名',loc='left',pad=9)
    ax.legend(frameon=False,loc='lower right',fontsize=7);sty(ax,'x')
    ax=fig.add_subplot(gs[1,0]);tag(ax,'c')
    names=['Bagging','Random forest','Logistic regression','RBF-SVM']
    barpos=np.arange(4)
    ax.bar(barpos-.18,vv.loc[names,'fn'],width=.35,color=RED,label='漏判恶性 FN')
    ax.bar(barpos+.18,vv.loc[names,'fp'],width=.35,color=BLUE,label='误判良性 FP')
    ax.set_xticks(barpos,[CHN[x] for x in names],rotation=18,ha='right');ax.set_ylim(0,4.4);ax.set_ylabel('固定验证集误判个数');ax.legend(frameon=False,ncol=2,loc='upper right',fontsize=7)
    ax.set_title('同样 AUC 高，漏判数也可能不同',loc='left',pad=9);sty(ax)
    ax=fig.add_subplot(gs[1,1]);tag(ax,'d')
    s=rd('D_G_prediction_stability.csv').set_index('model').loc[['Single tree','Bagging','Random forest','Logistic regression','RBF-SVM']]
    ax.barh(range(len(s)),s.flip_rate*100,color=[COLORS[k] for k in s.index],height=.63)
    for i,q in enumerate(s.flip_rate*100):ax.text(q+.5,i,f'{q:.1f}%',va='center',fontsize=7.3)
    ax.set_yticks(range(len(s)),[CHN[x] for x in s.index]);ax.invert_yaxis();ax.set_xlim(0,30);ax.set_xlabel('类别翻转率（%）')
    ax.set_title('30 次重训：模型预测有多稳定？',loc='left',pad=9);sty(ax,'x')
    save(fig,'Fig3_公平比较与误判')

    # --- Figure 4 ---
    fig=plt.figure(figsize=(7,5.0));gs=fig.add_gridspec(2,2,wspace=.47,hspace=.60)
    ax=fig.add_subplot(gs[:,0]);tag(ax,'a')
    fes=['mean radius','mean perimeter','mean area','worst radius','worst perimeter','worst area','radius error','perimeter error','worst concave points','mean concave points']
    co=X[fes].corr();m=ax.imshow(co,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
    lab=[a.replace('mean ','M. ').replace('worst ','W. ').replace(' error',' err') for a in fes]
    ax.set_xticks(range(len(fes)),lab,rotation=60,ha='right',fontsize=6.4);ax.set_yticks(range(len(fes)),lab,fontsize=6.5)
    ax.set_title('部分形态特征高度相关',loc='left',pad=10)
    ax.spines[['top','bottom','left','right']].set_visible(False)
    cbar=fig.colorbar(m,ax=ax,fraction=.044,pad=.02);cbar.ax.tick_params(labelsize=7);cbar.set_label('Pearson r',fontsize=7)
    imp=rd('D_feature_importance.csv')
    ax=fig.add_subplot(gs[0,1]);tag(ax,'b')
    s=imp.nlargest(7,'gini_importance').iloc[::-1]
    ax.barh(s.feature.str.replace('worst ','W. ',regex=False).str.replace('mean ','M. ',regex=False),s.gini_importance,color=BLUE,height=.67)
    ax.set_title('Gini：树更爱拿谁来分裂？',loc='left',pad=10);ax.set_xlabel('不纯度重要性');sty(ax,'x')
    ax=fig.add_subplot(gs[1,1]);tag(ax,'c')
    s=imp.nlargest(7,'perm_auc_drop_mean').iloc[::-1]
    ax.barh(s.feature.str.replace('worst ','W. ',regex=False).str.replace('mean ','M. ',regex=False),s.perm_auc_drop_mean*100,
        xerr=s.perm_auc_drop_sd*100,color=ORANGE,height=.66,ecolor='#7d8792',capsize=2)
    ax.axvline(0,color=NAVY,lw=.7)
    ax.set_title('置换：单独打乱后 AUC 降多少？',loc='left',pad=10);ax.set_xlabel('AUC 减少（百分点）');sty(ax,'x')
    save(fig,'Fig4_相关特征与重要性')

    # --- Figure 5 ---
    fig=plt.figure(figsize=(7,5.05));gs=fig.add_gridspec(2,2,wspace=.43,hspace=.60)
    gb=rd('E_gradient_boosting_stages.csv')
    q=gb[gb.learning_rate==.1]
    ax=fig.add_subplot(gs[0,0]);tag(ax,'a')
    ax.plot(q.n_estimators,q.val_log_loss,color=ORANGE,marker='o',markersize=2.7,lw=1.8,label='验证集')
    ax.plot(q.n_estimators,q.train_log_loss,color=BLUE,marker='s',markersize=2.1,lw=1.6,label='训练集')
    ax.set_ylim(0,.62);ax.set_xlabel('迭代轮数');ax.set_ylabel('Log-loss（越低越好）')
    ax.axvline(75,color=GRAY,lw=1,ls='--');ax.text(90,.34,'验证损失\n先降后升',color=ORANGE,fontsize=8)
    ax.legend(frameon=False,loc='upper right');ax.set_title('Boosting 不是训练越久越好',loc='left',pad=9);sty(ax)
    ax=fig.add_subplot(gs[0,1]);tag(ax,'b')
    for lr,color in [(.03,BLUE),(.10,ORANGE),(.30,GREEN)]:
     q=gb[gb.learning_rate==lr]; ax.plot(q.n_estimators,q.val_log_loss,lw=1.7,color=color,marker='o',markersize=2.5,label=f'lr = {lr:g}')
    ax.set_ylim(0,.65);ax.set_xlabel('迭代轮数');ax.set_ylabel('验证 Log-loss')
    ax.set_title('学习率越大，通常越早需要停',loc='left',pad=9);ax.legend(frameon=False);sty(ax)
    ax=fig.add_subplot(gs[1,0]);tag(ax,'c')
    for lr,color in [(.03,BLUE),(.10,ORANGE),(.30,GREEN)]:
     q=gb[gb.learning_rate==lr];ax.plot(q.n_estimators,q.val_auc,lw=1.7,color=color,marker='o',markersize=2.5,label=f'lr = {lr:g}')
    ax.set_ylim(.91,1.004);ax.set_xlabel('迭代轮数');ax.set_ylabel('验证 ROC-AUC');ax.set_title('AUC 变化小，不代表概率没问题',loc='left',pad=9);sty(ax)
    ax=fig.add_subplot(gs[1,1]);tag(ax,'d')
    ada=rd('E_adaboost_weight_trace.csv');q=ada[ada.stage==1];wrong=q[q.wrong_this_stage==1]
    right=q[q.wrong_this_stage==0]
    items=pd.concat([wrong.head(2),right.head(2)],ignore_index=True)
    xs=np.arange(len(items));ax.bar(xs-.18,items.weight_before,width=.36,color=BLUE,label='更新前')
    ax.bar(xs+.18,items.weight_after,width=.36,color=ORANGE,label='更新后')
    ax.set_xticks(xs,[('错分' if v else '正确')+'\n'+s.replace('WDBC-','') for v,s in zip(items.wrong_this_stage,items.sample_id)],fontsize=7.4)
    ax.set_ylabel('样本权重');ax.legend(frameon=False,loc='upper right',fontsize=7)
    ax.set_title('AdaBoost：难题被赋予更大权重',loc='left',pad=9);sty(ax)
    save(fig,'Fig5_Boosting训练动态')

    print('Wrote',len(list(OUT.glob('Fig*.png'))),'PNG multi-panel figures')
    if ci_rows: print(pd.DataFrame(ci_rows).to_string(index=False))


if __name__ == "__main__":
    main()
