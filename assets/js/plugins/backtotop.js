$(window).scroll(function() {
  var backToTopButton = $(".back-to-top");
  if ($(window).scrollTop() >= 200) {
    backToTopButton.css("bottom", "0px");
  } else {
    backToTopButton.attr('style', '');
  }
  backToTopButton.click(function() {
    $('body,html').stop().animate({
      scrollTop: 0
    }, "slow");
  });
  return false;
});