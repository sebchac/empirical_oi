
/* ---- Tarea 2 ---- */

cd "C:\Users\Paola Bordon\Dropbox\Métodos IO Empírica\tareas\tareaBLP\Tarea 2\"

use OTC_Data.dta, clear											       
															        		       
* Generate market share												       			      	    

gen share=sales/count													       			
bysort week store (brand): egen inshare=sum(share)									       
gen outshare=1-inshare													       
															       
* Generate Hausman instruments

sort store week brand
bysort week brand (store): egen totprice=sum(price)
bysort week brand (store): egen nmarket=count(price)
gen ivhausman=(totprice-price)/(nmarket-1)
drop totprice nmarket
sort store week brand 

* ---- Logit estimation ----

* OLS

gen y=log(share/outshare)
regress y price prom, cluster(store)
matrix ols=e(b)
matrix ols_price=ols[1,"price".."prom"]'
gen elas_ols=_b[price]*price*(1-share)

* OLS with brand dummies

regress y price prom i.brand, cluster(store)
matrix ols_brand=e(b)
matrix ols_price_brand=ols_brand[1,"price".."prom"]'
gen elas_ols_brand=_b[price]*price*(1-share)

* OLS with brand-store dummies

regress y price prom i.brand#i.store, cluster(store)
matrix ols_brand_store=e(b)
matrix ols_price_brand_store=ols_brand_store[1,"price".."prom"]'
gen elas_ols_brand_store=_b[price]*price*(1-share)

* IV. cost shifters

ivregress 2sls y (price=cost) prom, first cluster(store)
matrix iv_cost=e(b)
matrix iv_price_cost=iv_cost[1,"price".."prom"]'
gen elas_iv_cost=_b[price]*price*(1-share)

* IV with brand dummies

ivregress 2sls y (price=cost) prom i.brand, first cluster(store)
matrix iv_brand=e(b)
matrix iv_price_brand=iv_brand[1,"price".."prom"]'
gen elas_iv_brand=_b[price]*price*(1-share)

* IV with brand-store dummies

ivregress 2sls y (price=cost) prom i.brand#i.store, cluster(store)
matrix iv_brand_store=e(b)
matrix iv_price_brand_store=iv_brand_store[1,"price".."prom"]'
gen elas_iv_brand_store=_b[price]*price*(1-share)

* IV. Hausman instruments

ivregress 2sls y (price=ivhausman) prom, first cluster(store)
matrix iv_haus=e(b)
matrix iv_price_haus=iv_haus[1,"price".."prom"]'
gen elas_iv_haus=_b[price]*price*(1-share)

* IV with brand dummies

ivregress 2sls y (price=ivhausman) prom i.brand, first cluster(store)
matrix iv_haus2=e(b)
matrix iv_price_haus2=iv_haus2[1,"price".."prom"]'
gen elas_iv_haus2=_b[price]*price*(1-share)

* IV with brand-store dummies

ivregress 2sls y (price=ivhausman) prom i.brand#i.store, cluster(store)
matrix iv_haus3=e(b)
matrix iv_price_haus3=iv_haus3[1,"price".."prom"]'
gen elas_iv_haus3=_b[price]*price*(1-share)

* Coefficients
matrix results_ols=ols_price,ols_price_brand,ols_price_brand_store
matrix results_iv_cost=iv_price_cost,iv_price_brand,iv_price_brand_store
matrix results_iv_haus=iv_price_haus,iv_price_haus2,iv_price_haus3

matrix list results_ols
matrix list results_iv_cost
matrix list results_iv_haus

* Elasticities
tabstat el*, by(brand)

* ---- Nested logit estimation ----

gen segment=0
replace segment=1 if brand==1 |brand==2 |brand==3
replace segment=2 if brand==4 |brand==5 |brand==6
replace segment=3 if brand==7 |brand==8 |brand==9
replace segment=4 if brand==10 |brand==11

bysort week store segment: egen share_segm=total(share)
gen sjg=share/share_segm

gen l_sjg=log(share)-log(share_segm)
bysort week store segment: egen n_segm=count(share)

regress y price l_sjg prom, cluster(store)

ivregress 2sls y l_sjg (price =ivhausman n_segm) prom, first cluster(store)
matrix nl_iv_haus=e(b)
matrix nl_iv_price_haus=nl_iv_haus[1,"price".."prom"]' \nl_iv_haus[1,"l_sjg"]

* Cannot estimate modelo with brand dummies becuase of multicolineality with n_segm

drop elas*
matwrite using OTC_logit, matrix(ols_price ols_price_brand ///
                                  iv_price_cost iv_price_brand ///
                                  iv_price_haus iv_price_haus2 ///
                                  nl_iv_price_haus) replace

log close
exit
