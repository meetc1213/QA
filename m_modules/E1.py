'''Contains code for linear coeffs and some initial work'''
def grads():
    nl_grad_NN_to_CO = round(np.dot(Z_diff(NN, CO), AG_NN),3)
    l_grad_NN_to_CO = round(np.dot(d_Z_lambda(NN,CO,7/3,lam), AG_NN),3)

    nl_grad_NN_to_BF = round(np.dot(Z_diff(NN, BF), AG_NN),3)
    l_grad_NN_to_BF = round(np.dot(d_Z_lambda(NN,BF,7/3,lam), AG_NN),3)

    # calculating linear and non-linear energy gradients
    nl_grad_CO_to_NN = round(np.dot(Z_diff(CO, NN), AG_CO),3)
    l_grad_CO_to_NN = round(np.dot(d_Z_lambda(CO,NN,7/3,lam), AG_CO),3)

    nl_grad_CO_to_BF = round(np.dot(Z_diff(CO, BF), AG_CO),3)
    l_grad_CO_to_BF = round(np.dot(d_Z_lambda(CO,BF,7/3,lam), AG_CO),3)

    # calculating linear and non-linear energy gradients
    nl_grad_BF_to_NN = round(np.dot(Z_diff(BF, NN), AG_BF),3)
    l_grad_BF_to_NN = round(np.dot(d_Z_lambda(BF,NN,7/3,lam), AG_BF),3)

    nl_grad_BF_to_CO = round(np.dot(Z_diff(BF, CO), AG_BF),3)
    l_grad_BF_to_CO = round(np.dot(d_Z_lambda(BF,CO,7/3,lam), AG_BF),3)

    return nl_grad_NN_to_CO, l_grad_NN_to_CO, nl_grad_NN_to_BF, \
        l_grad_NN_to_BF, nl_grad_CO_to_NN, l_grad_CO_to_NN, \
            nl_grad_CO_to_BF, l_grad_CO_to_BF, nl_grad_BF_to_NN, \
                l_grad_BF_to_NN, nl_grad_BF_to_CO, l_grad_BF_to_CO

def lin_coeff():
    '''linearizing coeffecient prediction calculated at CO and BF'''
    # calculating linear (non linear z) and non-linear (linear z)energy gradients

    nl_grads= np.array([nl_grad_CO_to_NN, nl_grad_CO_to_BF,
                        nl_grad_BF_to_NN, nl_grad_BF_to_CO])

    l_grads= np.array([l_grad_CO_to_NN, l_grad_CO_to_BF,
                        l_grad_BF_to_NN, l_grad_BF_to_CO])

    # nl_grads = np.append(nl_grads,[nl_grad_NN_to_BF, nl_grad_NN_to_CO])
    # l_grads = np.append(l_grads,[l_grad_NN_to_BF, l_grad_NN_to_CO])
    C = round(np.mean(l_grads / nl_grads),3)

    NN_pred = ['NN','-', e_NN + C *nl_grad_NN_to_CO ,  e_NN + C *nl_grad_NN_to_BF]
    CO_pred = ['CO',e_CO + C *nl_grad_CO_to_NN,'-' ,  e_CO + C *nl_grad_CO_to_BF]
    BF_pred = ['BF',e_BF + C *nl_grad_BF_to_NN,e_BF + C *nl_grad_BF_to_CO,'-' ]

    data = [
        ['From \ To', 'NN', 'CO','BF'],
        ['NN',e_NN, e_NN + C *nl_grad_NN_to_CO ,  e_NN + C *nl_grad_NN_to_BF],
        ['CO',round(e_CO + C *nl_grad_CO_to_NN,3),e_CO ,  round(e_CO + C *nl_grad_CO_to_BF,3)],
        ['BF',round(e_BF + C *nl_grad_BF_to_NN,3),round(e_BF + C *nl_grad_BF_to_CO,3),e_BF ]
    ]
    table = generate_table(data)
    print('Using average of linearizing coeffecient at CO and BF')
    print()
    print(table)

def nl_grad():

    print('Predictions from NN, CO and BF just using non-linear energy gradient')
    print()
    data = [
        ['From \ To', 'NN','err','CO','err','BF','err'],
        ['NN', e_NN, 0,e_NN + nl_grad_NN_to_CO,round(e_NN + nl_grad_NN_to_CO - e_CO,3),e_NN + nl_grad_NN_to_BF,round(e_NN + nl_grad_NN_to_BF - e_BF,3)],
        ['CO',e_CO + nl_grad_CO_to_NN,round(e_CO + nl_grad_CO_to_NN - e_NN,3),e_CO,0, e_CO + nl_grad_CO_to_BF,round(e_CO + nl_grad_CO_to_BF-e_BF,3)],
        ['BF',e_BF + nl_grad_BF_to_NN,round(e_BF + nl_grad_BF_to_NN - e_NN,3),e_BF + nl_grad_BF_to_CO,round(e_BF + nl_grad_BF_to_CO - e_CO,3),e_BF,0 ]
    ]
    table = generate_table(data)
    print(table)
    print(f'NN actual energy = {e_NN}, \nCO actual energy = {e_CO}, \nBF actual energy = {e_BF}, \n')

def hessian():
    '''Using Hessian'''
    # modelling ax^2 + bx + c
    a = -0.139 - 3.126
    b = 0
    c = -132.748
    def app(l):
        return round(a*l**2 + b*l + c,3)

    print('Prediction of energies via the Hessian')
    print()
    data = [['Mol \ Method', 'Actual (PBE; unc-ccpvdz with RKS)','Hessian from paper','Descrepency'],
            [
                'NN',e_NN, app(0),e_NN-app(0),
            ],
            [
                'CO',e_CO, app(1),round(e_CO-app(1),3),
            ],
            [
                'BF',e_BF, app(2),round(e_BF-app(2),3),
            ]

            ]
    table = generate_table(data)
    print(table)

def lin_int_Z_matrix():
    '''Matrix with linear interpolation in Z'''
    print('Predictions from NN, CO and BF just using linear Z / non-linear energy gradient')
    print()
    data = [
        ['From \ To', 'NN','CO','BF'],
        ['NN', e_NN,e_NN + nl_grad_NN_to_CO,e_NN + nl_grad_NN_to_BF],
        ['CO',e_CO + nl_grad_CO_to_NN,e_CO, e_CO + nl_grad_CO_to_BF],
        ['BF',e_BF + nl_grad_BF_to_NN,e_BF + nl_grad_BF_to_CO,e_BF]
    ]
    table = generate_table(data)
    print(table)

def non_lin_Z_matrix():
    '''Matrix with linear interpolation in Z'''
    print('Predictions from NN, CO and BF just using non-linear Z / linear energy gradient between pairs')
    print()
    data = [
        ['From \ To', 'NN','CO','BF'],
        ['NN', e_NN,e_NN + l_grad_NN_to_CO,round(e_NN + l_grad_NN_to_BF,3)],
        ['CO',e_CO + l_grad_CO_to_NN,e_CO, e_CO + l_grad_CO_to_BF],
        ['BF',round(e_BF + l_grad_BF_to_NN,3),e_BF + l_grad_BF_to_CO,e_BF]
    ]
    table = generate_table(data)
    print(table)


def g_idea():
    '''Giorgio's idea'''
    A_N1  = AG_NN[0]*3/7*7**(-4/3)
    E_CO_pred = e_NN + A_N1*( 6**(7/3)+8**(7/3)-2*7**(7/3))
    return E_CO_pred

def generate_table(data):
    # Determine the maximum width of each column
    column_widths = [max(len(str(row[i])) for row in data) for i in range(len(data[0]))]

    # Generate the table header
    table = generate_table_row(data[0], column_widths)
    table += generate_table_row(['-' * width for width in column_widths], column_widths)

    # Generate the table rows
    for row in data[1:]:
        table += generate_table_row(row, column_widths)

    return table

def generate_table_row(row_data, column_widths):
    row = '|'
    for i, item in enumerate(row_data):
        row += f' {str(item):{column_widths[i]}} |'
    row += '\n'
    return row