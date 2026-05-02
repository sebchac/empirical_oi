/******************************************/
/**********     Pauta Tarea 2     *********/
/******************************************/

use "C:\Users\Paola Bordon\Dropbox\Métodos IO Empírica\tareas\Tarea 3 2023\ChileAnalysis.dta", clear

/***************** Pregunta 1 ****************/

/*** Create variables in logs ***/

gen q=ln(routput)
gen k=ln(K)
gen m=ln(M)
gen l=ln(L)
gen re=ln(N)
gen rs=ln(S)
gen v=ln(rva)
gen ri=ln(rinvest)

tsset id year 

/******************************************************/
/*** Estimation of gross output production function ***/
/******************************************************/

/*************** Pregunta 2 ************/

/*** Olley Pakes estimation, using investment as proxy ***/

opreg q, exit(cexit) state(k) proxy(ri) free(l) 
gen beta_l_ri=_coef[l] /*0.6930964*/
gen beta_k_ri=_coef[k] /*0.438976*/

/*************** Pregunta 3 **************/

opreg q if ciiu_3d==321, exit(cexit) state(k) proxy(ri) free(l) 
gen beta_l_ri_321 =_coef[l] /*0.7386171*/
gen beta_k_ri_321 =_coef[k] /*0.3415496*/


/*************** Pregunta 4 ***************/

/*** Olley Pakes estimation, using materials as proxy ***/

opreg q, exit(cexit) state(k) proxy(m) free(l) 
gen beta_l_mat =_coef[l]
gen beta_k_mat =_coef[k]

tab beta_l_mat /*0.2513489*/
tab beta_k_mat /*0.2910807*/

/*************** Pregunta 5 **************/

opreg q if ciiu_3d==321, exit(cexit) state(k) proxy(m) free(l) 
gen beta_l_mat_321 =_coef[l]
gen beta_k_mat_321 =_coef[k]

tab beta_l_mat_321 /*0.3068298*/
tab beta_k_mat_321 /*0.2232461*/

/**************** Pregunta 6 **************/

* Las estimaciones usando la inversión tienen muchos missing.

/*************** Pregunta 7 ***************/

/*** Levinson and Petrin, using materials as proxy  ***/

levpet q, free(l) proxy(m) capital(k) revenue
predict omega, omega

/**************** Pregunta 8 ***************/

levpet q if ciiu_3d==321, free(l) proxy(m) capital(k) revenue
predict omega_321, omega

gen pos_output=routput
replace pos_output=0 if routput<0

bysort year ciiu_3d: egen total_output=total(pos_output) 
bysort year: gen mkt_share=pos_output/total_output

gen wtomega_321=. 
levelsof year, local(anio)
	qui foreach x of local anio{
		summarize omega_321 [w=mkt_share] if year == `x' & ciiu_3d==321
		replace wtomega_321 = r(mean) if year == `x' & ciiu_3d==321
	}

tsset id year 
gen wtomegalag_321=L.wtomega_321 if ciiu_3d==321
bysort year: egen wtomega_lag_321=mode(wtomegalag_321) if ciiu_3d==321
gen omega_gro_321=(wtomega_321-wtomega_lag_321)/wtomega_lag_321 if ciiu_3d==321

line wtomega_321 year if ciiu_3d==321, ytitle(" ") title("Productivity by LP, Textile Industry") 
line omega_gro_321 year if ciiu_3d==321, ytitle(" ") title("Productivity Growth by LP, Textile Industry") 

/**************** Pregunta 9 ***************/

/*** Estimation of value added production function ***/
/*****************************************************/

/*** Olley Pakes estimation, using materials as proxy ***/

opreg v if ciiu_3d==321, exit(cexit) state(k) proxy(m) free(l) 
gen beta_l_mat_va_321 =_coef[l]
gen beta_k_mat_va_321 =_coef[k]
tab beta_l_mat_va_321 /*0.6776256*/
tab beta_k_mat_va_321 /*0.2451513*/
 
/**************** Pregunta 10 ***************/

/*** Levinson and Petrin ***/

levpet v if ciiu_3d==321, free(l) proxy(m) capital(k) 
gen beta_l_lp_va_321 =_coef[l]
gen beta_k_lp_va_321 =_coef[k]
tab beta_l_lp_va_321  /*0.6776256*/
tab beta_k_lp_va_321  /*0.2201069*/

/**************** Pregunta 11 ***************/

acfest v if ciiu_3d==321, state(k age) proxy(m) free(l) va second
predict omega_321_acf, omega



 




