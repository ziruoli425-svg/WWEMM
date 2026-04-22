
clear
use data.dta, clear
encode city,gen(City)

global Y GPP_MinMax  
global X a_MinMax b_MinMax  CO2_MinMax Tmean_MinMax Rn_MinMax i.year i.City     
global D PRCPTOT_MinMax
set seed 42 
ddml init partial, kfolds(5) 
ddml E[D|X]: pystacked $D $X, type(reg) method(lassocv)
ddml E[Y|X]: pystacked $Y $X, type(reg) method(lassocv)
ddml crossfit
ddml estimate,robust

ddml E[D|X]: pystacked $D $X, type(reg) method(gradboost)
ddml E[Y|X]: pystacked $Y $X, type(reg) method(gradboost)
ddml E[D|X]: pystacked $D $X, type(reg) method(rf)  
ddml E[Y|X]: pystacked $Y $X, type(reg) method(rf) 
ddml E[D|X]: pystacked $D $X, type(reg) method(nnet)
ddml E[Y|X]: pystacked $Y $X, type(reg) method(nnet)
ddml E[D|X]: pystacked $D $X, type(reg) method(svm)
ddml E[Y|X]: pystacked $Y $X, type(reg) method(svm)
ddml E[Y|X]: pystacked $Y $X, type(reg) method(elasticcv)
ddml E[D|X]: pystacked $D $X, type(reg) method(elasticcv)


est sto result1
**# Bookmark #15
est sto result2
est sto result3
outreg2 [result1 result2 result3]using word.doc,replace bdec(3) sdec(3) rdec(3) ctitle(A) keep( * ) addtext(city FE, YES,year FE, YES) //输出标准误


**************************
///稳健性检验
//缩尾处理
clear
use data.dta, clear
winsor2 sdd $X, replace cuts(5 95)
global Y A  
global X tem co2 srad pressure light wind i.year i.City    
global D water
set seed 42 
ddml init partial, kfolds(5)
ddml E[D|X]: pystacked $D $X, type(reg) method(lassocv)
ddml E[Y|X]: pystacked $Y $X, type(reg) method(lassocv)
ddml crossfit
ddml estimate, robust


//考虑省份时间交互
clear
use data.dta, clear
encode provin,gen(Provin)
global Y A   
global X tem co2 srad pressure light wind i.year i.City i.year#i.City
global D water
set seed 42 
ddml init partial, kfolds(5)
ddml E[D|X]: pystacked $D $X, type(reg) method(lassocv)
ddml E[Y|X]: pystacked $Y $X, type(reg) method(lassocv)
ddml crossfit
ddml estimate, robust
est sto result9
outreg2 [result9]using word.doc,replace bdec(3) sdec(3) rdec(3) ctitle(A) keep( * ) addtext(city FE, YES,year FE, YES) //输出标准误


//改变样本分割比例
clear
use data.dta, clear
drop in 1568/l
encode city,gen(City)
global Y A  
global X c1 c2 c3 c4 c5 c6 i.year i.City     
global D TXx
set seed 42 
ddml init partial, kfolds(3)
ddml init partial, kfolds(8)
ddml E[D|X]: pystacked $D $X, type(reg) method(lassocv)
ddml E[Y|X]: pystacked $Y $X, type(reg) method(lassocv)
ddml crossfit
ddml estimate, robust
est sto result4
est sto result5
outreg2 [result4 result5]using word.doc,replace bdec(3) sdec(3) rdec(3) ctitle(A) keep( * ) addtext(city FE, YES,year FE, YES) //输出标准误

//更换机器学习方法(套索、梯度提升、神经网络、支持向量机、弹性网络)
clear
use data.dta, clear
drop in 1568/l
encode city,gen(City)
global Y A  
global X c1 c2 c3 c4 c5 c6 i.year i.City     
global D TXx
set seed 42 
ddml init partial, kfolds(5)
ddml E[D|X]: pystacked $D $X, type(reg) method(nnet)
ddml E[Y|X]: pystacked $Y $X, type(reg) method(nnet)
ddml E[D|X]: pystacked $D $X, type(reg) method(svm)
ddml E[Y|X]: pystacked $Y $X, type(reg) method(svm)
ddml E[Y|X]: pystacked $Y $X, type(reg) method(elasticcv)
ddml E[D|X]: pystacked $D $X, type(reg) method(elasticcv)
ddml crossfit
ddml estimate, robust
est sto result6
est sto result7
est sto result8
outreg2 [result6 result7 result8]using word.doc,replace bdec(3) sdec(3) rdec(3) ctitle(A) keep( * ) addtext(city FE, YES,year FE, YES) //输出标准误


///异质性分析:ph dem provin
clear
use data.dta, clear
egen x = mean(dem)  
gen DEM=1 if dem>x
replace DEM = 0 if DEM == .    
keep if inlist(DEM,1)
keep if inlist(DEM,0)

clear
use data.dta, clear
egen xx = mean(ph)  
gen PH=1 if ph>xx
replace PH = 0 if PH == .    
keep if inlist(PH,1)
keep if inlist(PH,0)

clear
use data.dta, clear
drop in 1568/l
encode city,gen(City)
encode provin, gen(provin_id)
label list provin_id
keep if inlist(provin_id,4,5,2)
keep if inlist(provin_id,1,3,6)


global Y A  
global X c1 c2 c3 c4 c5 c6 i.year i.City     
global D TXx
set seed 42 
ddml init partial, kfolds(5)
ddml E[D|X]: pystacked $D $X, type(reg) method(lassocv)
ddml E[Y|X]: pystacked $Y $X, type(reg) method(lassocv)
ddml crossfit
ddml estimate, robust
est sto result10
outreg2 [result10]using word.doc,replace bdec(3) sdec(3) rdec(3) ctitle(A) keep( * ) addtext(city FE, YES,year FE, YES) 



