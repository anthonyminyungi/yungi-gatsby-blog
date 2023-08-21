module.exports = {
  title: `DeveLife`,
  description: `Develop software, Develop myself, Develop life.`,
  author: `[Anthony min]`,
  introduction: `Develop software, Develop myself, Develop life.`,
  siteUrl:
    process.env.NODE_ENV === 'development'
      ? 'http://localhost:8000'
      : `https://yungis.dev`, // Your blog site url
  social: {
    twitter: `dope_a_minn`, // Your Twitter account
    github: `anthonyminyungi`, // Your GitHub account
    medium: ``, // Your Medium account
    facebook: `dbstnsdl12`, // Your Facebook account
    instagram: `yungis_`,
  },
  icon: `content/assets/felog.png`, // Add your favicon
  keywords: [
    `blog`,
    `javascript`,
    `js`,
    `web`,
    `react`,
    `gatsby`,
    `frontend`,
    `develop`,
    `yungi`,
    `develife`,
    `develop`,
  ],
  comment: {
    disqusShortName: '', // Your disqus-short-name. check disqus.com.
    utterances: 'anthonyminyungi/yungi-gatsby-blog', // Your repository for archive comment
  },
  configs: {
    countOfInitialPost: 10, // Config your initial count of post
  },
  sponsor: {
    buyMeACoffeeId: 'anthonymin',
  },
  share: {
    facebookAppId: '2682139905214006', // Add facebookAppId for using facebook share feature v3.2
  },
  ga: 'UA-155331130-1', // Add your google analytics tranking ID
  ad: 'pub-8634316481995713', // Add your google adsense ID
}
