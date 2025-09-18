def test_imports():
    import backend.app.main as main
    assert hasattr(main, 'app')
