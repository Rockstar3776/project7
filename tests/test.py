from api.main import get_models_results



def test_the_first():
    assert 1==1

def test_return():
    resultat = get_models_results(214010)
    assert isinstance(resultat, str)