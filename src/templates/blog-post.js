import React, { useEffect } from 'react'
import { graphql } from 'gatsby'
import ReactDOM from 'react-dom'

import * as Elements from '../components/elements'
import { Layout } from '../layout'
import { Head } from '../components/head'
import { PostTitle } from '../components/post-title'
import { PostDate } from '../components/post-date'
import { PostContainer } from '../components/post-container'
import { SocialShare } from '../components/social-share'
import { SponsorButton } from '../components/sponsor-button'
import { Bio } from '../components/bio'
import { PostNavigator } from '../components/post-navigator'
import { Disqus } from '../components/disqus'
import { Utterences } from '../components/utterances'
import * as ScrollManager from '../utils/scroll'

import '../styles/code.scss'

export default ({ data, pageContext, location }) => {
  useEffect(() => {
    ScrollManager.init()
    // const codeTitles = document.getElementsByClassName('gatsby-code-title')
    // const macSvgIcon = (
    //   <svg
    //     xmlns="http://www.w3.org/2000/svg"
    //     width="54"
    //     height="14"
    //     viewBox="0 0 54 14"
    //   >
    //     <g fill="none" fill-rule="evenodd" transform="translate(1 1)">
    //       <circle
    //         cx="6"
    //         cy="6"
    //         r="6"
    //         fill="#FF5F56"
    //         stroke="#E0443E"
    //         stroke-width=".5"
    //       ></circle>
    //       <circle
    //         cx="26"
    //         cy="6"
    //         r="6"
    //         fill="#FFBD2E"
    //         stroke="#DEA123"
    //         stroke-width=".5"
    //       ></circle>
    //       <circle
    //         cx="46"
    //         cy="6"
    //         r="6"
    //         fill="#27C93F"
    //         stroke="#1AAB29"
    //         stroke-width=".5"
    //       ></circle>
    //     </g>
    //   </svg>
    // )

    // for (let t of codeTitles) {
    //   console.log(t)
    //   t.prepend(macSvgIcon)
    // }
    return () => ScrollManager.destroy()
  }, [])

  const post = data.markdownRemark
  const metaData = data.site.siteMetadata
  const { title, comment, siteUrl, author, sponsor } = metaData
  const { disqusShortName, utterances } = comment

  return (
    <Layout
      showToc={post.frontmatter.showToc}
      location={location}
      title={title}
    >
      <Head title={post.frontmatter.title} description={post.excerpt} />
      <PostTitle title={post.frontmatter.title} />
      <PostDate date={post.frontmatter.date} />
      <PostContainer html={post.html} />
      <SocialShare title={post.frontmatter.title} author={author} />
      {!!sponsor.buyMeACoffeeId && (
        <SponsorButton sponsorId={sponsor.buyMeACoffeeId} />
      )}
      <Elements.Hr />
      <Bio />
      <PostNavigator pageContext={pageContext} />
      {!!disqusShortName && (
        <Disqus
          post={post}
          shortName={disqusShortName}
          siteUrl={siteUrl}
          slug={pageContext.slug}
        />
      )}
      {!!utterances && <Utterences repo={utterances} />}
    </Layout>
  )
}

export const pageQuery = graphql`
  query BlogPostBySlug($slug: String!) {
    site {
      siteMetadata {
        title
        author
        siteUrl
        comment {
          disqusShortName
          utterances
        }
        sponsor {
          buyMeACoffeeId
        }
      }
    }
    markdownRemark(fields: { slug: { eq: $slug } }) {
      id
      excerpt(pruneLength: 280, truncate: true)
      html
      frontmatter {
        title
        date(formatString: "YYYY/MM/DD")
        showToc
      }
    }
  }
`
