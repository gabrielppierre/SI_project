import connection as cn

def main():
    port = 2037
    s = cn.connect(port)

    if s != 0:
        print("Conexão estabelecida com sucesso!")
    else:
        print("Falha ao estabelecer a conexão.")

if __name__ == "__main__":
    main()
